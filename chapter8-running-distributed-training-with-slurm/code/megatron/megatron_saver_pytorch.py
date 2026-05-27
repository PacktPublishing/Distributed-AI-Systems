"""
Simple PyTorch saver for Megatron checkpoint converter.

This saver can be used with convert.py to export Megatron checkpoints to standard PyTorch format.

Usage with convert.py:
    cd /home/wukong/workspace/coderepo/Megatron-LM/tools/checkpoint
    python convert.py \
        --model-type GPT \
        --loader core \
        --saver pytorch \
        --load-dir /path/to/megatron/checkpoint \
        --save-dir /path/to/output \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1
"""

import os
import sys
import torch
from collections import OrderedDict

# Add this directory to path so convert.py can find it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def add_arguments(parser):
    """Add command-line arguments for PyTorch saver."""
    group = parser.add_argument_group(title='PyTorch saver')
    group.add_argument('--output-filename', type=str, default='model.pt',
                       help='Output filename for PyTorch checkpoint')


class PyTorchSaver:
    """Saver that exports to standard PyTorch format."""
    
    def __init__(self, args, queue):
        self.args = args
        self.queue = queue
        self.md = None
        self.state_dict = OrderedDict()
        
    def save_checkpoint(self):
        """Main save function called by convert.py."""
        print("PyTorch saver started...")
        
        # Receive metadata
        metadata = self.queue.get()
        if metadata == "exit":
            print("Loader exited with error")
            return
        
        self.md = metadata
        print(f"Received metadata: model_type={self.md.model_type}, num_layers={self.md.num_layers}")
        
        # Receive embeddings
        embeddings = self.queue.get()
        if embeddings == "exit":
            return
        print(f"Received embeddings: {embeddings['name']}")
        self.state_dict.update({k: v for k, v in embeddings.items() if k != 'name'})
        
        # Receive transformer layers
        for layer_idx in range(self.md.num_layers):
            layer = self.queue.get()
            if layer == "exit":
                return
            print(f"Received layer {layer_idx}: {layer['name']}")
            # Add layer prefix
            layer_dict = {f"layers.{layer_idx}.{k}": v for k, v in layer.items() if k != 'name'}
            self.state_dict.update(layer_dict)
        
        # Receive final layer norm
        final_norm = self.queue.get()
        if final_norm == "exit":
            return
        print(f"Received final norm: {final_norm['name']}")
        self.state_dict.update({k: v for k, v in final_norm.items() if k != 'name'})
        
        # Receive LM head
        lm_head = self.queue.get()
        if lm_head == "exit":
            return
        print(f"Received LM head: {lm_head['name']}")
        self.state_dict.update({k: v for k, v in lm_head.items() if k != 'name'})
        
        # Wait for "done"
        done = self.queue.get()
        if done != "done":
            print(f"Unexpected message: {done}")
            return
        
        # Save checkpoint
        os.makedirs(self.args.save_dir, exist_ok=True)
        output_path = os.path.join(self.args.save_dir, getattr(self.args, 'output_filename', 'model.pt'))
        
        checkpoint = {
            'model_state_dict': self.state_dict,
            'model_config': {
                'model_type': self.md.model_type,
                'num_layers': self.md.num_layers,
                'hidden_size': self.md.hidden_size,
                'num_attention_heads': self.md.num_attention_heads,
                'vocab_size': self.md.true_vocab_size if hasattr(self.md, 'true_vocab_size') else None,
                'max_position_embeddings': self.md.max_position_embeddings,
                'seq_length': self.md.seq_length,
                'params_dtype': str(self.md.params_dtype),
            }
        }
        
        torch.save(checkpoint, output_path)
        file_size_gb = os.path.getsize(output_path) / 1e9
        print(f"✓ Saved PyTorch checkpoint to {output_path}")
        print(f"  File size: {file_size_gb:.2f} GB")
        print(f"  Total parameters: {sum(p.numel() for p in self.state_dict.values()) / 1e9:.2f}B")


def save_checkpoint(queue, args):
    """Required top-level function for convert.py."""
    saver = PyTorchSaver(args, queue)
    saver.save_checkpoint()


if __name__ == '__main__':
    print("This module is designed to be used with convert.py")
    print("See README_EXPORT.md for usage instructions")
