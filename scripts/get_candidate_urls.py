from datasets import load_dataset
from langdetect import detect, DetectorFactory, LangDetectException
from langdetect import detector_factory

def get_views(x):
    s = x["views"]
    for key in ["views", "plays", "play"]:
        s = s.replace(key, "").strip()
    try:
        x["views"] = float(s)
        return x
    except Exception as e:
        pass
    try:
        if s[-1] == "K":
            x["views"] = float(s[:-1]) * 1e3
        if s[-1] == "M":
            x["views"] = float(s[:-1]) * 1e6
        if s[-1] == "B":
            x["views"] = float(s[:-1]) * 1e9
        return x
    except Exception as e:
        pass
    print(f"Cannot map : {x['views']}")
    x["views"] = -1
    return x

DetectorFactory.seed = 0

def detect_language(x):
    try:
        language = detect(x["title"])
        return language in ["en", "zh-cn", "zh-tw", "ja", "ko"]
    except LangDetectException:
        return False

def main():
    ds = load_dataset("laion/LAION-DISCO-12M")
    mapped_ds = ds["train"].map(get_views, num_proc=8)
    filtered_ds = mapped_ds.filter(lambda x: x["views"] > 500000, num_proc=8)
    filtered_ds = filtered_ds.filter(lambda x: x["isExplicit"] is False, num_proc=8)
    filtered_ds = filtered_ds.filter(lambda x: x["duration"] < 600 and x["duration"] > 120, num_proc=8)
    filtered_ds = filtered_ds.filter(lambda x: len(x["artist_ids"]) > 0, num_proc=8)
    filtered_ds = filtered_ds.filter(lambda x: detect_language(x))
    filtered_ds = filtered_ds.remove_columns(["isExplicit"])
    filtered_ds.save_to_disk("./resources/dataset/filtered_ds")

if __name__ == "__main__":
    main()
