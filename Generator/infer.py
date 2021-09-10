import os
import logging
import random
from argparse import ArgumentParser
from pprint import pformat
from tqdm import tqdm

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, GPT2LMHeadModel, BertTokenizer, OpenAIGPTConfig, CONFIG_NAME, GPT2Config

SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "[BOS]", "[EOS]", "[POS]", "[NEG]", "[speaker1]", "[speaker2]"]


def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def build_input_from_segments(sequence_list, reply_list, tokenizer, with_eos=True):
    """ Build a sequence of input from 3 segments: sent, senti and last reply """

    all_input_ids = []
    all_input_index = []
    for sequence, reply in zip(sequence_list, reply_list):
        input_ids = tokenizer.convert_tokens_to_ids(sequence) + reply
        all_input_ids.append(input_ids)
        all_input_index.append(len(input_ids) - 1)

    padding_max_len = max(all_input_index) + 1

    all_input_ids = [input_ids + [0] * (padding_max_len - len(input_ids)) for input_ids in all_input_ids]
    all_attention_mask = [[1] * (index + 1) + [0] * (padding_max_len - index - 1) for index in all_input_index]
    all_token_type_ids = [[0] * padding_max_len for _ in all_input_index]

    return all_input_ids, all_attention_mask, all_token_type_ids, all_input_index


def handle_input(sent, senti, target_senti):
    """Handle input by different sent"""
    if senti == "[NEU]":
        h_words = sent.split(" ")
        del_num = max(1, round(len(h_words) * 25 / 100))
        mask_id = random.sample(range(0, len(h_words)), del_num)
        new_sent = []
        for i in range(len(h_words)):
            if i not in mask_id:
                new_sent.extend(list(h_words[i]))
        sequence = ["[BOS]"] + new_sent + ["[SEP]"]
    else:
        senti = target_senti
        sep_token = sent.split("\t")[1]
        new_sent = sent.split("\t")[-1].split(" ")

        sequence = ["[BOS]"] + [sep_token] + [senti] + ["[SEP]"] + new_sent + ["[SEP]"]

    return sequence, new_sent


def test_data(args):
    dataset = []
    file = args.datapath.split("=")
    for i in range(len(file)):
        with open(file[i], "r", encoding="utf-8") as f:
            dataset.extend(f.read().splitlines())
    return dataset


def sample_sequence(sequence_list, tokenizer, model, args):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    current_output_list = [[] for _ in sequence_list]

    finish_index_list = []
    for i in range(args.max_length):
        input_ids, attention_mask, token_type_ids, input_index = build_input_from_segments(sequence_list, current_output_list, tokenizer, with_eos=False)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=args.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=args.device)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=args.device)
        input_index = torch.tensor(input_index, dtype=torch.long, device=args.device)
        input_range = torch.arange(input_index.shape[0], dtype=torch.long, device=args.device)

        logits, *_ = model(input_ids, attention_mask, token_type_ids)
        logits = logits[input_range, input_index, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs_ = F.softmax(logits, dim=-1)

        for j in range(probs_.shape[0]):
            if j in finish_index_list:
                continue

            probs = probs_[j]
            prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)

            if i < args.min_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    prev = torch.multinomial(probs, num_samples=1)

            if prev.item() in special_tokens_ids:
                finish_index_list.append(j)
                continue

            current_output_list[j].append(prev.item())

        if len(finish_index_list) >= len(sequence_list):
            break

    return current_output_list


def main():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for test")
    parser.add_argument("--senti", type=str, default="[NEU]", help="Senti: [NEU] [SENTI] [SENTI-NEU] [SENTI-WITH-NEU] [SENTI-NEU-SEP]")
    parser.add_argument("--target_senti", type=str, default="[POS]", help="Target Senti: [NEU] [POS] [NEU]")
    parser.add_argument('--gpt2', action='store_true', help="use gpt2")
    parser.add_argument("--datapath", type=str, default="", help="Path of the dataset.")
    parser.add_argument("--out_path", type=str, default="", help="Path of response generated.")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of subprocesses for data loading")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=30, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.0, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        logging.error("Checkpoint needed!")
        return

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    if os.path.isfile(args.model_checkpoint):
        model_checkpoint = os.path.split(args.model_checkpoint)[0]
    else:
        model_checkpoint = args.model_checkpoint
    tokenizer_class = BertTokenizer
    model_class = OpenAIGPTLMHeadModel if not args.gpt2 else GPT2LMHeadModel
    config_class = OpenAIGPTConfig if not args.gpt2 else GPT2Config
    tokenizer = tokenizer_class.from_pretrained(model_checkpoint, do_lower_case=True)
    config = config_class.from_json_file(os.path.join(model_checkpoint, CONFIG_NAME))
    model = model_class.from_pretrained(args.model_checkpoint, config=config)

    model.to(args.device)
    model.eval()

    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    dataset = test_data(args)

    logger.info("\tModel_path: %s" % args.model_checkpoint)
    logger.info("\tTest_path: %s" % args.datapath)
    logger.info("\tOut_path: %s" % args.out_path)
    logger.info("\tData_size: %d" % len(dataset))

    # print(dataset[0:3])

    def inner_loop(sent_list, sequence_list, new_sent_list):
        predictions = []
        res = []

        with torch.no_grad():
            out_ids_list = sample_sequence(sequence_list, tokenizer, model, args)

        for sent, out_ids, new_sent in zip(sent_list, out_ids_list, new_sent_list):
            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            if args.senti == "[SENTI-WITH-NEU]":
                res.append("{}\t{}".format(sent, " ".join(new_sent)))
                res.append("{}\t{}".format(sent, out_text))
            elif args.senti == "[SENTI-NEU-SEP]":
                res.append("{}\t{}".format(sent, out_text))
            elif args.senti == "[SENTI-NEU]":
                res.append("{}\t{}".format(sent, out_text))
            elif args.senti == "[NEU-WITH-SENTI]":
                res.append("{}\t{}".format(sent, out_text))
            elif args.senti == "[SENTI]":
                res.append("{}\t{}\t{}".format(sent, " ".join(new_sent), out_text))
            else:
                predictions.append("{}；{}；{}".format(sent, " ".join(new_sent), out_text))

        return predictions, res

    logger.info("\t====SENTI: %s====" % args.senti)
    logger.info("\t====TARGET_SENTI: %s====" % args.target_senti)

    sent_list = []
    sequence_list = []
    new_sent_list = []
    for sent_ in tqdm(dataset, mininterval=1):
        sequence, new_sent = handle_input(sent_, senti=args.senti, target_senti=args.target_senti)
        sent_list.append(sent_)
        sequence_list.append(sequence)
        new_sent_list.append(new_sent)

    logger.info("\t====Infer Example====")
    for i in range(5):
        logger.info("\t====Example %d====" % i)
        logger.info("\tOrigial Sent: " + sent_list[i])
        logger.info("\tInput Sequence: " + " ".join(sequence_list[i]))

    batch_sent_list = [sent_list[i:i + args.batch_size] for i in range(0, len(sent_list), args.batch_size)]
    batch_sequence_list = [sequence_list[i:i + args.batch_size] for i in range(0, len(sequence_list), args.batch_size)]
    batch_new_sent_list = [new_sent_list[i:i + args.batch_size] for i in range(0, len(new_sent_list), args.batch_size)]

    all_predictions = []
    all_res = []
    for sent_list, sequence_list, new_sent_list in tqdm(zip(batch_sent_list, batch_sequence_list, batch_new_sent_list), total=len(batch_sent_list)):
        predictions, res = inner_loop(sent_list, sequence_list, new_sent_list)
        all_predictions.append(predictions)
        all_res.append(res)

    predictions = [y for x in all_predictions for y in x]
    res = [y for x in all_res for y in x]

    if args.senti == "[NEU]":
        with open(args.out_path, 'w', encoding="UTF-8") as f:
            f.write("\n".join(predictions))
    else:
        logger.info("   Res_size: %d" % len(res))
        with open(args.out_path, 'w', encoding="UTF-8") as f:
            f.write("\n".join(res))


if __name__ == "__main__":
    main()
