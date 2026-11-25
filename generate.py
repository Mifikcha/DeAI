import argparse
import codecs
from pathlib import Path

import torch

from model import LanguageModel, ModelConfig


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Continue a UTF-8 prompt")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/observations.pt"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-bytes", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7, help="0 for greedy decoding")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.prompt or args.max_new_bytes < 0 or not args.temperature >= 0:
        parser.error("use a nonempty prompt, nonnegative byte count and temperature")
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = LanguageModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rng = torch.Generator().manual_seed(args.seed)
    output = bytearray(args.prompt.encode("utf-8"))
    for _ in range(args.max_new_bytes):
        context = torch.tensor(list(output[-model.config.context:])).unsqueeze(0)
        logits = model(context)[0, -1]
        if args.temperature == 0:
            token = logits.argmax()
        else:
            probs = torch.softmax((logits - logits.max()) / args.temperature, dim=0)
            token = torch.multinomial(probs, 1, generator=rng)
        output.append(token.item())
    # A byte budget can end partway through the final UTF-8 character.
    print(codecs.getincrementaldecoder("utf-8")(errors="replace").decode(output))


if __name__ == "__main__":
    main()
