import re
from typing import Callable, Literal

PUNCTUATION = set(["!", "?", "…", ",", ".", "-", " "])
PUNCTUATIONS = "".join(re.escape(p) for p in PUNCTUATION)

SPLITS = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}

METHODS: dict[str, Callable] = dict()


class TextProcessor:
    TEXT_SPLIT_METHODS: dict[str, Callable[[str], str]] = dict()

    def __init__(self) -> None:
        pass

    @property
    def method_names(self) -> list[str]:
        return list(self.TEXT_SPLIT_METHODS.keys())

    @staticmethod
    def replace_consecutive_punctuation(text: str) -> str:
        """Replace consecutive punctuation

        Args:
            text (str): Input Texts

        Returns:
            str: Output Texts
        """
        pattern = f"([{PUNCTUATIONS}])([{PUNCTUATIONS}])+"
        result = re.sub(pattern, r"\1", text)
        return result

    @staticmethod
    def split_long_text(text: str, max_len=510) -> list[str]:
        """Split long texts for bert input

        Args:
            text (str): Input Texts
            max_len (int, optional): Max Lens For Input Texts. Defaults to 510.

        Returns:
            list[str]: Output Texts
        """
        punctuation = "".join(SPLITS)

        segments = re.split(f"([{punctuation}])", text)

        result: list[str] = []
        current_segment = segments[0]

        for segment in segments[1:]:
            if len(current_segment + segment) > max_len:
                result.append(current_segment)
                current_segment = segment
            else:
                current_segment += segment

        if current_segment:
            result.append(current_segment)

        return result

    @staticmethod
    def split(inp_text: str) -> list[str]:
        """Split By punctuation

        Args:
            inp_text (str): Input Texts

        Returns:
            list[str]: Output Texts
        """
        inp_text = inp_text.replace("……", ".").replace("——", ",")
        if inp_text[-1] not in SPLITS:
            inp_text += "."
        i_split_head = i_split_tail = 0
        len_text = len(inp_text)
        opt_texts: list[str] = []
        while 1:
            if i_split_head >= len_text:
                break
            if inp_text[i_split_head] in SPLITS and (
                not (inp_text[i_split_head - 1].isdigit() and inp_text[max(i_split_head + 1, len_text)].isdigit())
            ):
                i_split_head += 1
                opt_texts.append(inp_text[i_split_tail:i_split_head])
                i_split_tail = i_split_head
            else:
                i_split_head += 1
        return opt_texts

    @staticmethod
    def register_method(name: str):
        def decorator(func: Callable):
            TextProcessor.TEXT_SPLIT_METHODS[name] = func
            return func

        return decorator

    @register_method("cut0")  # Do Nothing
    @staticmethod
    def cut0(inp: str):
        """Almost Do nothing

        Args:
            inp (str): Input Texts

        Returns:
            str: Output Texts
        """
        if not set(inp).issubset(PUNCTUATION):
            return inp
        else:
            return "\n"

    @register_method("cut1")
    @staticmethod
    def cut1(inp: str) -> str:
        """Cut per 4 sentence

        Args:
            inp (str): Input Texts

        Returns:
            str: Output Texts
        """
        inp = inp.strip("\n")
        inps = TextProcessor.split(inp)
        split_idx = list(range(0, len(inps), 4)) + [len(inps)]
        if len(split_idx) > 1:
            opts = []
            for idx in range(len(split_idx) - 1):
                opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
        else:
            opts = [inp]
        opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
        return "\n".join(opts)

    @register_method("cut2")
    @staticmethod
    def cut2(inp: str) -> str:
        """Cut per 50 letters

        Args:
            inp (str): Input Texts

        Returns:
            str: Output Texts
        """
        inp = inp.strip("\n")
        inps = TextProcessor.split(inp)
        if len(inps) < 2:
            return inp
        opts = []
        summ = 0
        tmp_str = ""
        for i in range(len(inps)):
            summ += len(inps[i])
            tmp_str += inps[i]
            if summ > 50:
                summ = 0
                opts.append(tmp_str)
                tmp_str = ""
        if tmp_str != "":
            opts.append(tmp_str)
        if len(opts) > 1 and len(opts[-1]) < 50:
            opts[-2] = opts[-2] + opts[-1]
            opts = opts[:-1]
        opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
        return "\n".join(opts)

    @register_method("cut3")
    @staticmethod
    def cut3(inp: str) -> str:
        """Cut by fullwidth period

        Args:
            inp (str): Input Texts

        Returns:
            str: Output Texts
        """
        inp = inp.strip("\n")
        opts = ["%s" % item for item in inp.strip("。").split("。")]
        opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
        return "\n".join(opts)

    @register_method("cut4")
    @staticmethod
    def cut4(inp: str) -> str:
        """Cut by halfwidth period

        Args:
            inp (str): Input Texts

        Returns:
            str: Output Texts
        """
        inp = inp.strip("\n")
        opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
        opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
        return "\n".join(opts)

    # contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
    @register_method("cut5")
    @staticmethod
    def cut5(inp: str) -> str:
        """Cut By Punctuations

        Args:
            inp (str): Input Texts

        Returns:
            str: Output Texts
        """
        inp = inp.strip("\n")
        punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
        mergeitems = []
        items = []

        for i, char in enumerate(inp):
            if char in punds:
                if char == "." and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                    items.append(char)
                else:
                    items.append(char)
                    mergeitems.append("".join(items))
                    items = []
            else:
                items.append(char)

        if items:
            mergeitems.append("".join(items))

        opt = [item for item in mergeitems if not set(item).issubset(punds)]
        return "\n".join(opt)

    @staticmethod
    def cut(inp: str, method_name: Literal["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"]) -> str:
        """Cut text using given method

        Args:
            inp (str): Input Texts
            method_name (Literal[&quot;cut0&quot;, &quot;cut1&quot;, &quot;cut2&quot;, &quot;cut3&quot;, &quot;cut4&quot;, &quot;cut5&quot;]): Text Split Method

        Returns:
            str: Output Texts

        Raises:
            RuntimeError: Unknown Method
        """
        method = METHODS.get(method_name, None)
        if method is None:
            raise RuntimeError(f"Unknown Method {method_name}")
        return method(inp)
