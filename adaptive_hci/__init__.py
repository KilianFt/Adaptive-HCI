from common import GeneralModelEnum
from adaptive_hci.controllers import EMGViT, ContextClassifier

GENERAL_MODELS = {
    GeneralModelEnum.CClassifier: ContextClassifier,
    GeneralModelEnum.ViT: EMGViT,
}
