import json
from google.cloud import translate

PARENT = f"projects/zeta-rush-412601"

def translate_text(text: str, target_language_code: str) -> translate.Translation:
    client = translate.TranslationServiceClient()

    response = client.translate_text(
        parent=PARENT,
        contents=[text],
        target_language_code=target_language_code,
    )

    return response.translations[0].translated_text

def translate_task_file(in_file, out_file):
    with open(in_file, "r") as f:
        data = json.load(f)

    texts = [j for i in data for j in i["utterances"]]
    translated_text = {}
    for text in texts:
        translated_text[text] = translate_text(text, "en")

    json_object = json.dumps(translated_text, indent=4)
    with open(out_file, "w") as outfile:
        outfile.write(json_object)

translate_task_file("task1.json", "translated_task1.json")