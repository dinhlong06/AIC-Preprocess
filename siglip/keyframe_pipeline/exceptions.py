class KeyframePipelineError(Exception):
    """Base exception for all public pipeline failures."""


class InvalidArgumentError(KeyframePipelineError):
    """A public argument is internally inconsistent or out of range."""


class DatasetLayoutError(KeyframePipelineError):
    """The dataset directory layout is unsupported."""


class InvalidKeyframeInputError(KeyframePipelineError):
    """A keyframe record or image is invalid."""


class ModelConfigurationError(KeyframePipelineError):
    """A configured model/backend cannot satisfy the public contract."""


class ModelInferenceError(KeyframePipelineError):
    """A model failed or returned an invalid response."""


class ArtifactValidationError(KeyframePipelineError):
    """A serialized result does not match its schema/invariants."""


class OutputAlreadyExistsError(KeyframePipelineError):
    """A target exists while overwrite is disabled."""
