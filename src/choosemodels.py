import numpy as np

def is_integerish(arr, tol=1e-6):
    arr = np.asarray(arr, float)
    return np.allclose(arr, np.round(arr), atol=tol)

def choose_models(models, y_train, cuenta,
                  last_k=6, min_nonzero_last=3,
                  HIGH_ZERO=0.60, VERY_HIGH_ZERO=0.80):
    """
    Devuelve un diccionario con los modelos a entrenar para 'cuenta'
    usando TU 'models' actual (las claves que existan).
    """
    y = np.asarray(y_train, float).ravel()
    zero_ratio = (y == 0).mean()
    has_neg    = np.any(y < 0)

    # señales recientes
    y_last = y[-last_k:] if len(y) >= last_k else y
    nonzero_last = int(np.sum(y_last != 0))
    # racha de no-ceros al final
    nz_streak = 0
    for v in y[::-1]:
        if v != 0: nz_streak += 1
        else: break
    revived = (nonzero_last >= min_nonzero_last) or (nz_streak >= min_nonzero_last)

    selected = {}

    if has_neg:
        #  fuera ZIP y Tweedie (Lightgbm Tweedie)
        # ✔ TwoPart con etiqueta (y != 0)
        if "TwoPart" in models: selected["TwoPart"] = models["TwoPart"]
        # ✔ Regresores “puros”
        for k in ["Ridge","RidgePSO","Lasso","LassoPSO","Linear","LinearPSO"]:
            if k in models: selected[k] = models[k]
        # ✔ LGBM en modo regresión si lo tienes
        if "Lightgbm_reg" in models: selected["Lightgbm_reg"] = models["Lightgbm_reg"]

    else:
        # No hay negativos
        base_regs = {}
        for k in ["Ridge","RidgePSO","Lasso","LassoPSO","Linear","LinearPSO"]:
            if k in models: base_regs[k] = models[k]
        if "Lightgbm" in models: base_regs["Lightgbm"] = models["Lightgbm"]  # Tweedie OK si y>=0
        
        if zero_ratio >= HIGH_ZERO:
            # Muchos ceros
            # TwoPart (y>0) si la tienes; si no, usa la que ya tienes
            if "TwoPart_pos" in models:
                selected["TwoPart"] = models["TwoPart_pos"]
            elif "TwoPart" in models:
                selected["TwoPart"] = models["TwoPart"]

            # ZIP solo si integer-ish
            if "ZeroInflatedPoisson" in models and is_integerish(y):
                selected["ZeroInflatedPoisson"] = models["ZeroInflatedPoisson"]

            # Tweedie (Lightgbm) si y>=0
            if "Lightgbm" in models:
                selected["Lightgbm"] = models["Lightgbm"]

            # Si la serie “revivió”, habilita también regresores puros
            if revived:
                selected.update(base_regs)

        else:
            # Pocos ceros → todo lo estándar
            selected.update(base_regs)
            if "TwoPart_pos" in models:
                selected["TwoPart"] = models["TwoPart_pos"]
            elif "TwoPart" in models:
                selected["TwoPart"] = models["TwoPart"]
            if "ZeroInflatedPoisson" in models and is_integerish(y):
                selected["ZeroInflatedPoisson"] = models["ZeroInflatedPoisson"]


    # ... si vas a considerar ZIP:
    if "ZeroInflatedPoisson" in models:
        if has_neg or (not is_integerish(y)) or (zero_ratio == 1.0):
            # NO lo agregues
            pass
        else:
            selected["ZeroInflatedPoisson"] = models["ZeroInflatedPoisson"]
    # fallback por si quedó vacío (muy raro)
    if not selected:
        for k in ["Ridge","Lightgbm","TwoPart","Lasso","Linear","RidgePSO","LinearPSO","LassoPSO"]:
            if k in models:
                selected[k] = models[k]; break

    print(f"[{cuenta}] zeros={zero_ratio:.1%} | neg={has_neg} | last{last_k}_nonzero={nonzero_last} | nz_streak={nz_streak} | revived={revived}")
    return selected
