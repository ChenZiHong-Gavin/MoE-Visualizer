# Adapted from https://github.com/QwenLM/Qwen2.5/blob/main/examples/demo/web_demo.py

from argparse import ArgumentParser
from threading import Thread

import gradio as gr
import torch
import json
import tempfile
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from moe_visualizer.plot_histogram import plot_histogram

DEFAULT_CKPT_PATH = "Qwen/Qwen1___5-MoE-A2___7B-Chat"


def _get_args():
    parser = ArgumentParser(description="Qwen1.5-MoE Visualizer Demo")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CKPT_PATH,
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=False,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=8000, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
    )

    args = parser.parse_args()
    return args


class ExpertActivationTracker:
    def __init__(self):
        self.activations = defaultdict(list)

    def add_activation(self, layer_idx, activation):
        self.activations[layer_idx].append(activation)

    def clear(self):
        self.activations.clear()


expert_tracker = ExpertActivationTracker()


@torch.no_grad()
def moe_activation_hook_factory(layer_idx: int):
    def hook_fn(module, input, output):
        hidden_states = input[0]
        _, _, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = module.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        _, selected_experts = torch.topk(routing_weights, module.top_k, dim=-1)  # (batch_size, num_tokens, top_k)

        expert_tracker.add_activation(layer_idx, selected_experts.cpu().numpy())

    return hook_fn


def _register_hooks(model):
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "gate"):
            hook = layer.mlp.register_forward_hook(moe_activation_hook_factory(layer_idx))
            hooks.append(hook)
    return hooks


def count_expert_activations():
    prefill_expert_counts = defaultdict(lambda: defaultdict(int))
    generate_expert_counts = defaultdict(lambda: defaultdict(int))

    for layer_idx, activations in expert_tracker.activations.items():
        for i, arr in enumerate(activations):
            if arr.shape[0] > 1:
                counts = prefill_expert_counts[layer_idx]
            else:
                counts = generate_expert_counts[layer_idx]
            for token in arr.flatten():
                counts[int(token)] += 1
    return prefill_expert_counts, generate_expert_counts


def prepare_data():
    prefill_expert_counts, generate_expert_counts = count_expert_activations()

    data = {
        "prefill_expert_counts": prefill_expert_counts,
        "generate_expert_counts": generate_expert_counts
    }

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(data, f)
        return f.name


def generate_plots():
    prefill_expert_counts, generate_expert_counts = count_expert_activations()

    prefill_fig = plot_histogram(prefill_expert_counts)
    generate_fig = plot_histogram(generate_expert_counts)
    return prefill_fig, generate_fig


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype="auto",
        device_map=device_map,
        resume_download=True,
        trust_remote_code=True
    ).eval()

    hooks = _register_hooks(model)
    model.hooks = hooks

    model.generation_config.max_new_tokens = 2048

    return model, tokenizer


def _chat_stream(model, tokenizer, query, history):
    conversation = []
    for query_h, response_h in history:
        conversation.append({"role": "user", "content": query_h})
        conversation.append({"role": "assistant", "content": response_h})
    conversation.append({"role": "user", "content": query})
    input_text = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


def _gc():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _launch_demo(args, model, tokenizer):
    def predict(_query, _chatbot):
        reset_state(_chatbot)

        print(f"User: {_query}")
        _chatbot.append({"role": "user", "content": _query})
        _chatbot.append({"role": "assistant", "content": ""})
        full_response = ""
        response = ""
        for new_text in _chat_stream(model, tokenizer, _query, history=[]):
            response += new_text
            _chatbot[-1] = {"role": "assistant", "content": response}
            yield _chatbot
            full_response = response

        print(f"Qwen: {full_response}")

    def process_batch(batch_file, progress=gr.Progress()):
        if not batch_file:
            raise gr.Error("No file uploaded")

        try:
            questions = []
            with open(batch_file, "r") as f:
                data = json.load(f)
                for item in data:
                    if "question" in item:
                        questions.append(item["question"])
        except Exception as e:
            raise gr.Error(f"Failed to parse file: {e}")

        if not questions:
            raise gr.Error("No question found in the file")

        progress(0, "Processing...")
        for i, question in enumerate(tqdm(questions)):
            for _ in _chat_stream(model, tokenizer, question, history=[]):
                pass
            progress((i + 1) / len(questions))

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot):
        _chatbot.clear()
        expert_tracker.clear()
        _gc()
        return _chatbot

    with gr.Blocks() as demo:
        gr.HTML("<h1 style='text-align: center'>Qwen1.5-MoE Visualizer</h1>")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Qwen1.5-MoE", type="messages", elem_classes="control-height")
                query = gr.Textbox(lines=2, label="Input")

                with gr.Row():
                    submit_btn = gr.Button("🚀 Submit (发送)")

                with gr.Group():
                    batch_file = gr.File(label="📤 Upload Batch File (JSON) (上传批处理文件)")
                    batch_btn = gr.Button("🔄 Execute Batch (执行批处理)")

            with gr.Column(scale=2):
                prefill_plot = gr.Plot(label="Expert Activation (Prefill)")
                generate_plot = gr.Plot(label="Expert Activation (Generate)")

                json_file = gr.File(visible=False, label="Download Data")

            submit_btn.click(
                predict, [query, chatbot], [chatbot], show_progress=True
            ).then(
                generate_plots, [], [prefill_plot, generate_plot]
            ).then(
                prepare_data, [], [json_file]
            ).then(
                lambda: gr.update(visible=True),
                outputs=[json_file]
            )
            submit_btn.click(reset_user_input, [], [query])

            batch_btn.click(
                process_batch,
                [batch_file],
                [],
                show_progress=True
            ).then(
                generate_plots,
                [],
                [prefill_plot, generate_plot]
            ).then(
                prepare_data,
                [],
                [json_file]
            )

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    model, tokenizer = _load_model_tokenizer(args)

    _launch_demo(args, model, tokenizer)


if __name__ == "__main__":
    main()
