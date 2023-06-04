import torch
import transformers

from tqdm import tqdm

from lm_eval import utils
from lm_eval.base import BaseLM


class GPTNeoLM(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="EleutherAI/gpt-neo-1.3B",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        padding="left",
        seed=0,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        torch.manual_seed(seed)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")

        self.gpt2 = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
            device_map="auto"
        ) #.to(self.device)
        self.gpt2.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
        )

        assert isinstance(
            self.tokenizer,
            (
                transformers.GPT2Tokenizer,
                transformers.GPT2TokenizerFast,
                transformers.T5Tokenizer,
                transformers.T5TokenizerFast,
                transformers.GPTNeoXTokenizerFast,
            ),
        ), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        if isinstance(
            self.tokenizer,
            (
                transformers.GPT2Tokenizer,
                transformers.GPT2TokenizerFast,
            )
        ):
            assert self.tokenizer.encode("hello\n\nhello") == [
                31373,
                198,
                198,
                31373,
            ], self.tokenizer.encode("hello\n\nhello")

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        self.tokenizer.padding_side = padding
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.gpt2.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 16

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, strings):

        tokenizer_args = {"add_special_tokens": False, "return_tensors": "pt"}

        if (type(strings) == list) or type(strings) == tuple:
            return self.tokenizer.batch_encode_plus(strings, padding=True, **tokenizer_args)
        elif type(strings) == str:
            return self.tokenizer.encode(strings, **tokenizer_args)

    def tok_decode(self, tokens):
        if len(tokens.shape) > 1:
            return self.tokenizer.batch_decode(tokens)
        else:
            return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.gpt2(inps)[0][:, :, :50257]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.gpt2.generate(
            **context, max_length=max_length, eos_token_id=eos_token_id, # do_sample=True,
        )

    def greedy_until(self, requests):
        # TODO: implement fully general `until` that handles until that are
        #       multiple tokens or that span multiple tokens correctly

        # TODO: extract to TokenizedLM?
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        disable_tqdm = False
        for chunk in utils.chunks(
            tqdm(re_ord.get_reordered(), disable=disable_tqdm), self.batch_size
        ):

            context, until = zip(*chunk)

            until = until[0]
            if isinstance(until, str):
                until = [until]

            (primary_until,) = self.tok_encode(until[0]).to(self.device)

            context_enc = self.tok_encode(context).to(self.device)

            if self.batch_size > 1:
                context_enc = {key: value[:, self.max_gen_toks - self.max_length :] for key, value in context_enc.items()}

            cont = self._model_generate(
                # **context_enc,
                context_enc,
                max_length=context_enc['input_ids'].shape[1] + self.max_gen_toks,
                eos_token_id=primary_until
            )

            s = self.tok_decode(cont[:, context_enc['input_ids'].shape[1]:])

            for term in until:
                s = [i.split(term)[0] for i in s]

            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)

            res.extend(s)

        return re_ord.get_original(res)