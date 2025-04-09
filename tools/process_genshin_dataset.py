import librosa
import argparse
import os
import shutil
from funasr import AutoModel
import soundfile as sf
from tqdm import tqdm
from tools.slicer2 import Slicer


def process_text(speaker, wav_path, lang):
    # wav_path = os.path.join(args.in_dir, speaker, wav_name)
    global speaker_annos
    tr_name = wav_path[:-4]
    with open(tr_name + ".lab", "r", encoding="utf-8") as file:
        text = file.read()
    text = text.replace("{NICKNAME}", '旅行者')
    text = text.replace("{M#他}{F#她}", '他')
    text = text.replace("{M#她}{F#他}", '他')
    text = text.replace("|", '')
    if "{M#妹妹}{F#哥哥}" in text:
        if tr_name.endswith("a"):
            text = text.replace("{M#妹妹}{F#哥哥}", '妹妹')
        if tr_name.endswith("b"):
            text = text.replace("{M#妹妹}{F#哥哥}", '哥哥')
    text = text.replace("#", '')
    text = f'{lang}|{text}\n'  #
    speaker_annos.append(f"{wav_path}|{speaker}|{text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default=os.getcwd(), help="path to source dir")
    args = parser.parse_args()
    path_asr = 'tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
    path_asr = path_asr if os.path.exists(path_asr) else "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    model_revision = "v2.0.4"
    path_vad = 'tools/asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch'
    path_punc = 'tools/asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
    path_vad = path_vad if os.path.exists(path_vad) else "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    path_punc = path_punc if os.path.exists(path_punc) else "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    vad_model_revision = punc_model_revision = "v2.0.4"
    # sync with gsv
    model = AutoModel(
        model=path_asr,
        model_revision=model_revision,
        vad_model=path_vad,
        vad_model_revision=vad_model_revision,
        punc_model=path_punc,
        punc_model_revision=punc_model_revision,
    )

    speaker_annos = []
    speaker_name = os.path.split(args.in_dir)[-1]
    lang = "ZH"
    file_list = [os.path.join(args.in_dir, i) for i in os.listdir(args.in_dir) if i.endswith(".wav")]
    for i in tqdm(file_list,total=len(file_list)):
        wav, sr = librosa.load(i, sr=None)
        if wav.shape[-1] > sr * 15:
            os.makedirs(os.path.join(args.in_dir,"backup"),exist_ok=True)
            shutil.move(i,os.path.join(args.in_dir,"backup",os.path.basename(i)))
            slicer = Slicer(sr=sr)
            count = 0
            for chunk, start, end in slicer.slice(wav):
                try:
                    sliced_path = f"{i[:-4]}_{count}.wav"
                    sf.write(sliced_path, chunk, sr)
                    count += 1
                    text = model.generate(input=sliced_path)[0]["text"]
                    with open(sliced_path[:-4] + ".lab", "w", encoding="utf-8") as f:
                        f.write(text)
                    process_text(speaker_name, sliced_path, lang)
                except Exception as e:
                    print(e)
                    continue
        else:
            process_text(speaker_name, i, lang)
    with open(os.path.join(os.path.dirname(args.in_dir), f"{speaker_name}.list"), 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)
    print("finished.")
