from models import timer, timer_xl, moirai, moment, gpt4ts, ttm, time_llm, autotimes, timerope, timer_rope, timer_timerope, timer_wo_pe, moment_rope, moment_timerope, moment_wo_pe


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "timer": timer,
            "timer_xl": timer_xl,
            "moirai": moirai,
            "moment": moment,
            "gpt4ts": gpt4ts,
            "ttm": ttm,
            "time_llm": time_llm,
            "autotimes": autotimes,
            "timerope": timerope,
            "timer_rope": timer_rope,
            "timer_timerope": timer_timerope,
            "timer_wo_pe": timer_wo_pe,
            "moment_rope": moment_rope,
            "moment_timerope": moment_timerope,
            "moment_wo_pe": moment_wo_pe,
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
