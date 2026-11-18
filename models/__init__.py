from .dino import build_dino_model, dino_available_models, dino_default_gradcam_target_layer
from .openclip import build_openclip_model, openclip_available_models, openclip_default_gradcam_target_layer
from .retfound import build_retfound_model, retfound_available_models, retfound_default_gradcam_target_layer
from .mae import build_mae_model, mae_available_models, mae_default_gradcam_target_layer

available_models = dino_available_models \
                    + openclip_available_models \
                    + retfound_available_models \
                    + mae_available_models

def build_encoder(cfg):

    if cfg.enc_name in dino_available_models:
        return build_dino_model(cfg)

    elif cfg.enc_name in openclip_available_models:
        return build_openclip_model(cfg)

    elif cfg.enc_name in retfound_available_models:
        return build_retfound_model(cfg)

    elif cfg.enc_name in mae_available_models:
        return build_mae_model(cfg)

    else:
        raise NotImplementedError(f'{cfg.enc_name} not in {available_models}')


def get_default_gradcam_target_layer(model, cfg):
    if cfg.enc_name in dino_available_models:
        return dino_default_gradcam_target_layer(model)

    elif cfg.enc_name in openclip_available_models:
        return openclip_default_gradcam_target_layer(model)

    elif cfg.enc_name in retfound_available_models:
        return retfound_default_gradcam_target_layer(model)

    elif cfg.enc_name in mae_available_models:
        return mae_default_gradcam_target_layer(model)

    else:
        raise NotImplementedError(f'{cfg.enc_name} not in {available_models}')
