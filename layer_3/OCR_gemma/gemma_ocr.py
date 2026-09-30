"""OCR + caption keyframe bằng Gemma 4 (LLM UIT), lưu riêng -- không đụng output của ../OCR.

Vì sao 1120 token/ảnh: mặc định server chỉ cấp 280 token cho một ảnh 1280x720, chữ nhỏ
bị vỡ và Gemma đoán ra câu tiếng Việt trôi chảy nhưng sai ("ma túy" -> "máy bay",
"1961" -> "1978"). 1120 đọc đúng các ca đó; đổi lại chậm ~3.5x (~4.2 frame/s).
OCR và caption là 2 request riêng: gộp chung 1 request (JSON {ocr, caption}) nhanh hơn
~1.5x nhưng phần OCR sót ~8% chữ (hay bỏ dòng nhỏ cuối khung hình).

    F=../../layer_2/Keyframe_Extracting/benchmark_batch1_v2/pipeline_g
    python gemma_ocr.py --task ocr     --frames $F --out gemma_ocr_batch1.jsonl
    # caption: AI Studio chạy từ cuối, mỗi (key, model) một process một shard; UIT chạy từ đầu, bỏ qua phần đã làm
    python gemma_ocr.py --task caption --model ais26 --key 2 --reverse --shard 3/10 --frames $F --out ais26_k2_caption_batch1.jsonl --skip "*_caption_batch1.jsonl"
    python gemma_ocr.py --task caption --frames $F --out gemma_caption_batch1.jsonl --skip "*_caption_batch1.jsonl"
    python gemma_ocr.py --task ocr --frames $F --out sample.jsonl --sample 45 --copy-to ../../gemma_ocr_image
"""

import argparse, base64, glob, json, os, random, shutil, ssl, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Qwen chạy trên GPU riêng: dồn caption sang đó khi Gemma đang bận OCR. Cả hai đều nghẽn ở server chứ
# không ở số request (Gemma 1120: 16->40 song song chỉ 4.1->4.4 frame/s; Qwen ~1.3 frame/s ở 8 hay 16).
# ais26/ais31: Gemma 4 trên Google AI Studio (GEMINI_API_KEY) -- quota riêng, chạy song song với UIT. Ảnh cố
# định 258 token (mediaResolution bị bỏ qua) nhưng caption ngang UIT 1120 trên 6 ảnh khó. Quota free mỗi model
# 30 RPM / 16K TPM / 14.4K RPD, ~650 token/request -> tự giãn nhịp ~22 request/phút.
AIS_BASE = os.environ.get("GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent")
UIT_BASE = os.environ.get("UIT_BASE_URL", "https://llm.uit.edu.vn").rstrip("/")
AIS = AIS_BASE
MODELS = {"gemma": (f"{UIT_BASE}/gemma/v1/chat/completions", "gemma-4-26b", int(os.environ.get("GEMMA_WORKERS", 20))),
          "qwen": (f"{UIT_BASE}/qwen/v1/chat/completions", "qwen3.8-27b", 4),
          "ais26": (AIS.format("gemma-4-26b-a4b-it"), "gemma-4-26b-a4b-it", 12),
          "ais31": (AIS.format("gemma-4-31b-it"), "gemma-4-31b-it", 32)}  # url, model, workers
# Đo 2026-09-24 (26B, ảnh 578 token vào/request): 4 key x 20/phút = 77 caption/phút (97% thành công); 25/phút/key hoặc
# tổng gửi >~100/phút thì 429 tăng vọt và tổng caption/phút GIẢM. Chạy ~4 key x 20; đổi bằng biến môi trường AIS_RPM.
# Đo lại chiều 2026-09-24: 1 key riêng ~25/phút nhưng 10 key cùng lúc gần 0 -- trần chung nhiều key, thêm key không giúp.
AIS_RPM = float(os.environ.get("AIS_RPM", 20))
_pace = {"lock": threading.Lock(), "next": 0.0, "pause": 0.0}


def _wait_turn():
    # Luồng đã nhận lượt trước khi có 429 phải xếp lại sau lệnh nghỉ, không thì vẫn gửi trong lúc nghỉ.
    while True:
        with _pace["lock"]:
            t = max(time.time(), _pace["next"])
            _pace["next"] = t + 60 / AIS_RPM
        time.sleep(max(0.0, t - time.time()))
        if time.time() >= _pace["pause"]:
            return
PROMPT = """Bạn là công cụ OCR. Chép lại CHÍNH XÁC các dòng chữ nhìn thấy rõ trong ảnh, từ trên xuống dưới, mỗi dòng chữ một dòng.
Quy tắc:
- Chỉ chép ký tự thực sự nhìn thấy. KHÔNG đoán, KHÔNG suy luận nội dung, KHÔNG tự viết tiếp hay hoàn thành câu, KHÔNG sửa chính tả, KHÔNG dịch.
- Từ nào mờ, quá nhỏ hoặc bị che mà không chắc chắn thì BỎ từ đó; cả dòng không đọc chắc được thì BỎ cả dòng.
- Chữ lặp lại nhiều lần (logo, biển giống nhau) chỉ ghi một lần.
- Giữ nguyên dấu tiếng Việt đúng như trong ảnh.
- KHÔNG dùng "...", "[...]" hay bất kỳ ký hiệu nào thay cho chữ không đọc được — chỉ đơn giản bỏ qua.
- Không có chữ nào đọc chắc được thì trả về rỗng. Không giải thích, không thêm gì khác."""
CAPTION_PROMPT = """Viết mô tả khung hình video để người dùng tìm lại được ảnh này bằng cách gõ những gì họ nhìn thấy. 2-3 câu tiếng Việt.
- Nêu các đặc trưng nhìn thấy được: người (số lượng, nam/nữ, trẻ em/người lớn/người già, quần áo và màu), con vật, đồ vật (gọi đúng tên), hành động (động từ cụ thể), bối cảnh, địa điểm, vị trí (phía trước, phía sau, bên trái, bên phải), góc máy (cận cảnh, toàn cảnh, nhìn từ trên xuống).
- Gọi đúng tên khi nhận ra chắc chắn: đồ vật, món ăn, loài vật, địa danh, công trình, nhân vật nổi tiếng. Không chắc thì tả hình dáng, KHÔNG đoán tên.
- Chỉ dùng từ chỉ đặc điểm nhìn thấy được (màu sắc, số lượng, kích thước, hình dạng, chất liệu). KHÔNG dùng tính từ cảm thán, đánh giá hay cảm xúc (đẹp, rực rỡ, hiện đại, sang trọng, yên bình, tươi ngon, hấp dẫn, nổi bật...).
- KHÔNG dùng từ phỏng đoán (dường như, có thể, giống như, có vẻ, hoặc).
- BỎ QUA chữ, logo, watermark, phụ đề (đã có OCR riêng). KHÔNG viết câu phủ định. KHÔNG tả bầu không khí, cảm xúc, mục đích.
Chỉ trả về đoạn mô tả, không thêm gì khác."""
# Caption 560 (người dùng chọn để nhanh hơn ~1.7x): chấm tay 8 ảnh thì 1120 chính xác hơn -- 560 bịa 3-4 chi tiết ("bế một đứa trẻ" ở cảnh hai người mò
# dưới nước), 1120 gần như không -- caption dùng để khớp mô tả người dùng gõ nên chi tiết bịa kéo ảnh sai lên.
TASKS = {"ocr": (PROMPT, 1120, 768), "caption": (CAPTION_PROMPT, 560, 300)}  # prompt, token/ảnh, max_tokens
# Server trả 429 ngay khi vượt ~40 request đồng thời/key; key dùng chung với retrieval_system
# backend nên chừa chỗ cho nó -- Gemma 32 + Qwen 16 chạy cùng lúc vẫn chưa thấy 429.

# UIT endpoint presents a private CA, nhưng bundle là cấu hình deployment chứ không phải
# code nên không nằm trong repo. Trỏ UIT_CA_BUNDLE vào file đó; không đặt thì dùng
# trust store của hệ thống (đủ cho các model ais*, vốn gọi Google).
_ca_bundle = os.environ.get("UIT_CA_BUNDLE")
ctx = ssl.create_default_context(cafile=_ca_bundle) if _ca_bundle else None


def ask(path, task, model, key=1):
    prompt, budget, max_tokens = TASKS[task]
    url, name, _ = MODELS[model]
    img = base64.b64encode(path.read_bytes()).decode()
    if model.startswith("ais"):
        # thinkingLevel MINIMAL: Gemma trên AI Studio mặc định suy nghĩ ~200 token/ảnh, ăn vào quota TPM.
        # maxOutputTokens bị giữ chỗ trước trong TPM (300 thì 429 xen kẽ, <=150 thì không); caption dài nhất kho 163 token,
        # 40 ảnh chạy lại với 200 ra y hệt 300. OCR cũng vậy: p99 kết quả batch2 ~320 ký tự (~120 token),
        # để 768 thì mỗi request giữ chỗ TPM gấp ~6 lần cần thiết.
        max_tokens = 200
        body = {"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": img}}, {"text": prompt}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens, "thinkingConfig": {"thinkingLevel": "MINIMAL"}}}
        req = urllib.request.Request(f"{url}?key={os.environ['GEMINI_API_KEY' + (f'_{key}' if key > 1 else '')]}", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    else:
        # Qwen: tắt thinking (chậm ~10x, không cần cho mô tả ảnh); không nhận max_soft_tokens.
        extra = {"mm_processor_kwargs": {"max_soft_tokens": budget}} if model == "gemma" \
            else {"chat_template_kwargs": {"enable_thinking": False}}
        body = {"model": name, "max_tokens": max_tokens, "temperature": 0, "repetition_penalty": 1.1, **extra,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}},
                    {"type": "text", "text": prompt}]}]}
        req = urllib.request.Request(url, json.dumps(body).encode(),
                                     {"Authorization": "Bearer " + os.environ["LLM_API_KEY"], "Content-Type": "application/json"})
    for attempt in range(12 if model.startswith("ais") else 5):
        try:
            if model.startswith("ais"):
                _wait_turn()
                c = json.load(urllib.request.urlopen(req, timeout=180))["candidates"][0]
                # Phản hồi có 1 part "thought" (rỗng) + MỘT HOẶC NHIỀU part chữ: lấy part cuối từng làm mất phần
                # đầu caption ở ~13% ảnh của 31B -- phải ghép mọi part không phải thought.
                text = "".join(x.get("text", "") for x in c["content"]["parts"] if not x.get("thought"))
                return text.strip(), c.get("finishReason", "").lower()
            c = json.load(urllib.request.urlopen(req, context=ctx, timeout=300))["choices"][0]
            return (c["message"]["content"] or "").replace("<turn|>", "").strip(), c["finish_reason"]
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise
            if model.startswith("ais") and e.code == 429:
                # Request bị 429 vẫn bị tính vào hạn mức: lùi 10-30s thì retry nuôi chính nó, 4 key đứng 0 caption suốt
                # 3 tiếng (2026-09-24). Key chỉ hồi sau ~110s im lặng hẳn -> dừng cả process 120s.
                with _pace["lock"]:
                    _pace["pause"] = time.time() + 120
                    _pace["next"] = max(_pace["next"], _pace["pause"])
            print(f"{path.stem}: HTTP {e.code} {e.read()[:160]!r}", flush=True)
            time.sleep(min(2 ** attempt, 60))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"{path.stem}: {e!r}", flush=True)
            time.sleep(min(2 ** attempt, 60))
    # Bỏ qua thay vì ném lỗi: ảnh L25_V018_000002_kf0001 làm server treo >5 phút, từng khoá đầu hàng ghi kết quả
    # 9 tiếng và mất trắng. Không ghi dòng nào cho frame này nên chạy lại sẽ thử lại.
    print(f"{path.stem}: bỏ qua sau 5 lần lỗi", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--frames", help="thư mục <video_id>/<keyframe_id>.jpg (pipeline_g)")
    ap.add_argument("--images", help="glob ảnh jpg thay cho --frames; frame_id = tên file")
    ap.add_argument("--model", choices=MODELS, default="gemma")
    ap.add_argument("--out", required=True, help="jsonl, chạy lại sẽ bỏ qua frame đã có")
    ap.add_argument("--skip", nargs="*", default=[], help="jsonl (nhận glob) của job khác cùng task: bỏ qua frame đã có ở đó")
    ap.add_argument("--rpm", type=int, default=AIS_RPM, help="ais*: số request/phút mỗi process (giữ dưới quota RPM/TPM của key)")
    ap.add_argument("--key", type=int, default=1, help="ais*: dùng GEMINI_API_KEY_<n> (1 = GEMINI_API_KEY); mỗi key một project = quota riêng")
    ap.add_argument("--reverse", action="store_true", help="chạy từ cuối danh sách (2 job chia đôi từ hai đầu)")
    ap.add_argument("--range", help="a:b: chỉ lấy frame có chỉ số a..b-1 trong danh sách đã sắp (trước --reverse/--shard)")
    ap.add_argument("--shard", help="a,b/n: chỉ lấy frame có (chỉ số mod n) thuộc {a,b}; chia việc cho nhiều job cùng chiều")
    ap.add_argument("--sample", type=int, help="chỉ chạy N frame ngẫu nhiên (seed cố định)")
    ap.add_argument("--scan", help="thư mục scan/ của run_batch2.sh: chỉ lấy video có <video>.ok, bỏ ảnh trong <video>.bad")
    ap.add_argument("--copy-to", help="chép ảnh + .txt của từng frame vào đây để xem tay")
    args = ap.parse_args()
    globals()["AIS_RPM"] = args.rpm

    # Chỉ lấy video layer_2 đã ghi xong (có statistics.json): chạy song song với layer_2. Có --scan
    # thì chỉ lấy video đã quét và bỏ ảnh đọc bị treo (một luồng treo là process không bao giờ xong).
    # Video N* (camera giao thông batch2) bỏ ở đây: góc máy đứng yên, chữ chỉ là tên giao lộ + đồng hồ trên overlay
    # nên chỉ OCR frame đầu mỗi video (run_batch2.sh gemma_ocr_n, --images), ingest chép sang mọi keyframe.
    if args.images:
        frames = sorted(map(Path, glob.glob(args.images)))
    elif args.scan:
        scan = Path(args.scan)
        frames = sorted(f for ok in scan.glob("*.ok") if not ok.stem.startswith("N")
                        for bad in [set(ok.with_suffix(".bad").read_text().split())]
                        for f in (Path(args.frames) / ok.stem).glob("*.jpg") if f.name not in bad)
    else:
        frames = sorted(f for d in Path(args.frames).iterdir() if (d / "statistics.json").exists() for f in d.glob("*.jpg"))
    if args.sample:
        frames = sorted(random.Random(0).sample(frames, args.sample))
    if args.range:
        lo, hi = map(int, args.range.split(":"))
        frames = frames[lo:hi]
    if args.reverse:
        frames.reverse()
    if args.shard:
        idx, n = args.shard.split("/")
        keep = {int(i) for i in idx.split(",")}
        frames = [p for k, p in enumerate(frames) if k % int(n) in keep]
    out = Path(args.out)
    # Ổ /workingspace_aiclub (NFS dùng chung) hay treo I/O hàng chục phút khi máy tải cao (load ~390, nhiều process
    # người khác kẹt D-state cả ngày): luồng chính kẹt ở write() thì kết quả chỉ nằm trong RAM, kill process là mất.
    # Nên ghi vào spool CỤC BỘ trước, một luồng riêng chép sang file thật; chạy lại thì đọc cả hai để bỏ qua frame đã có.
    spool = Path("/tmp/gemma_spool") / out.name
    spool.parent.mkdir(exist_ok=True)
    out_ids = {json.loads(l)["frame_id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    done = {json.loads(l)["frame_id"] for pat in args.skip for f in glob.glob(pat) for l in open(f, encoding="utf-8")}
    done |= out_ids
    if spool.exists():
        done |= {json.loads(l)["frame_id"] for l in open(spool, encoding="utf-8") if l.endswith("\n")}
    todo = [p for p in frames if p.stem not in done]
    print(f"{len(frames)} frame, {len(done)} đã có, còn {len(todo)}", flush=True)

    sync_state = {"offset": 0}

    def sync_once():
        with open(spool, "rb") as sf:
            sf.seek(sync_state["offset"])
            chunk = sf.read()
        end = chunk.rfind(b"\n") + 1
        rows = [l for l in chunk[:end].decode("utf-8").splitlines() if json.loads(l)["frame_id"] not in out_ids]
        if rows:
            with out.open("a", encoding="utf-8") as of:
                of.write("\n".join(rows) + "\n")
                of.flush()
            out_ids.update(json.loads(l)["frame_id"] for l in rows)
        sync_state["offset"] += end

    def sync_loop():
        while True:
            time.sleep(15)
            try:
                sync_once()
            except OSError as e:
                print(f"sync sang {out.name} lỗi, thử lại sau: {e!r}", flush=True)

    spool.touch()
    threading.Thread(target=sync_loop, daemon=True).start()

    t0 = time.time()
    # as_completed, không phải ex.map: map trả kết quả THEO THỨ TỰ nên một request treo chặn mọi lần ghi phía sau.
    with spool.open("a", encoding="utf-8") as f, ThreadPoolExecutor(MODELS[args.model][2]) as ex:
        futs = {ex.submit(ask, p, args.task, args.model, args.key): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            p, res = futs[fut], fut.result()
            if res is not None:
                f.write(json.dumps({"frame_id": p.stem, "path": str(p), "text": res[0], "finish": res[1], "model": args.model + (f"#{args.key}" if args.key > 1 else "")},
                                   ensure_ascii=False) + "\n")
                f.flush()
            if i % 500 == 0:
                print(f"{i}/{len(todo)}  {i / (time.time() - t0):.1f} frame/s", flush=True)
    while True:
        try:
            sync_once()
            if sync_state["offset"] >= spool.stat().st_size:
                break
        except OSError as e:
            print(f"sync cuối lỗi, thử lại: {e!r}", flush=True)
            time.sleep(15)

    if args.copy_to:
        dst = Path(args.copy_to); dst.mkdir(parents=True, exist_ok=True)
        wanted = {p.stem for p in frames}
        rows = [r for r in map(json.loads, out.open(encoding="utf-8")) if r["frame_id"] in wanted]
        md = [f"# {args.model} {args.task} -- mẫu kiểm tra tay\n"]
        for r in sorted(rows, key=lambda r: r["frame_id"]):
            shutil.copy(r["path"], dst / f"{r['frame_id']}.jpg")
            (dst / f"{r['frame_id']}.txt").write_text(r["text"] + "\n", encoding="utf-8")
            md.append(f"## {r['frame_id']}\n\n![]({r['frame_id']}.jpg)\n\n```\n{r['text'] or '(rỗng)'}\n```\n")
        (dst / "README.md").write_text("\n".join(md), encoding="utf-8")
        print(f"-> {dst} ({len(rows)} ảnh)")


if __name__ == "__main__":
    main()
