import json
import locale
import os
import platform
from pathlib import Path
from typing import Optional

I18N_JSON_DIR: os.PathLike = os.path.join(os.path.dirname(os.path.relpath(__file__)), "locale")


def load_language_list(language):
    with open(os.path.join(I18N_JSON_DIR, f"{language}.json"), "r", encoding="utf-8") as f:
        language_list = json.load(f)
    return language_list


def scan_language_list() -> list[str]:
    language_list = []
    for name in os.listdir(I18N_JSON_DIR):
        if name.endswith(".json"):
            language_list.append(name.split(".")[0])
    return sorted(language_list)


class I18nAuto:
    def __init__(self, language: Optional[str] = None):
        if language in ["Auto", None]:
            language = locale.getlocale(6)[0]
            # getlocale can't identify the system's language ((None, None))
            if platform.system() == "Darwin" and language is None:
                try:
                    import plistlib

                    plist_path = Path("/Library/Preferences/.GlobalPreferences.plist")
                    if plist_path.exists():
                        with plist_path.open("rb") as plist_file:
                            plist_data: dict = plistlib.load(plist_file)
                            languages = plist_data.get("AppleLanguages", ["en_US"])
                            main_language_parts = languages[0].split("-")
                            main_language = main_language_parts[0].lower()
                            region = main_language_parts[-1].upper() if len(main_language_parts) > 1 else None

                            for lang in scan_language_list():
                                if region and lang == f"{main_language}_{region}":
                                    language = lang
                                    break
                            else:
                                for lang in scan_language_list():
                                    if lang.startswith(main_language + "_"):
                                        language = lang
                                        break
                finally:
                    pass
        if not os.path.exists(os.path.join(I18N_JSON_DIR, f"{language}.json")):
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key: str) -> str:
        return self.language_map.get(key, key)

    def __repr__(self):
        return "Use Language: " + self.language


if __name__ == "__main__":
    i18n = I18nAuto()
    print(i18n)
