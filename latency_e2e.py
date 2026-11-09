import os, sys, json, time, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
import tensorflow_datasets as tfds
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as prep_m
from tensorflow.keras.applications.resnet50 import preprocess_input as prep_r

ap = argparse.ArgumentParser()
ap.add_argument("--tau", type=float, required=True)
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--ref", default=None)
ap.add_argument("--out", default="pilot_artifacts/latency_e2e.json")
a = ap.parse_args()

def load_models():
    return (tf.keras.models.load_model("models_tmp/model_M_seed0.keras"),
            tf.keras.models.load_model("models_tmp/model_R_seed0.keras"))

def load_images(n):
    ds = tfds.load("cassava", split="test", as_supervised=True)
    return [(img.numpy(), int(l)) for img, l in ds.take(n)]

def pre(img, prep):
    x = tf.image.resize(img, (224, 224))
    return prep(tf.cast(x, tf.float32))[tf.newaxis]

def make_configs(mM, mR, tau):
    def fwd(m, x): return m(x, training=False).numpy()[0]
    def c_m(img):  return fwd(mM, pre(img, prep_m)).argmax(), 0
    def c_r(img):  return fwd(mR, pre(img, prep_r)).argmax(), 0
    def c_sv(img): return ((fwd(mM, pre(img, prep_m)) + fwd(mR, pre(img, prep_r))) / 2).argmax(), 1
    def c_casc(img):
        pM = fwd(mM, pre(img, prep_m))
        if pM.max() >= tau:
            return pM.argmax(), 0
        return ((pM + fwd(mR, pre(img, prep_r))) / 2).argmax(), 1
    return {"MobileNetV2": c_m, "ResNet50": c_r, "Soft voting": c_sv, "Cascade": c_casc}

def run(mM, mR, imgs, tau, warmup):
    cfg = make_configs(mM, mR, tau)
    names_ = list(cfg)
    for img, _ in imgs[:warmup]:                      # warm-up, not timed
        for f in cfg.values(): f(img)
    lat = {k: [] for k in names_}; pred = {k: [] for k in names_}; esc = []
    for i, (img, _) in enumerate(imgs):
        order = names_[i % len(names_):] + names_[:i % len(names_)]   # rotate the order
        for k in order:
            t0 = time.perf_counter(); p, e = cfg[k](img); dt = time.perf_counter() - t0
            lat[k].append(dt * 1000); pred[k].append(int(p))
            if k == "Cascade": esc.append(e)
    return lat, pred, esc

if __name__ == "__main__":
    mM, mR = load_models()
    imgs = load_images(a.n)
    lat, pred, esc = run(mM, mR, imgs, a.tau, a.warmup)
    res = {"n_images": len(imgs), "tau": a.tau, "threads": 2, "cpu_only": True,
           "escalation_rate_subset": float(np.mean(esc)), "configs": {}}
    for k, v in lat.items():
        v = np.array(v)
        res["configs"][k] = {"mean_ms": float(v.mean()), "median_ms": float(np.median(v)),
                             "p95_ms": float(np.percentile(v, 95)),
                             "latencies_ms": [round(float(x), 3) for x in v]}   # per image, for pooling runs
    if a.ref and os.path.exists(a.ref):
        ref = np.load(a.ref)
        res["cascade_agreement_with_saved_predictions"] = float(
            (np.array(pred["Cascade"]) == ref["cascade"][:len(imgs)]).mean())
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "configs"}
                     | {"configs": {k: {m: x for m, x in v.items() if m != "latencies_ms"}
                                    for k, v in res["configs"].items()}}, indent=2))
