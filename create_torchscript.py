import torch
import argparse

from codes.models.networks.sr_networks import RRDBNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', '-m', type=str, help = 'Python model to convert to \
        torchscript', choices = ['rrdb'], required = True)
    parser.add_argument('--model_state', '-s', type = str, help = 'Path to saved weights of the \
        model', required = True)
    parser.add_argument('--output_path', '-o', type = str, help = 'Path where torchscript model \
        is to be saved', required = True)
    parser.add_argument('--use_gpu', action = 'store_true')

    args = parser.parse_args()

    if args.model_type == 'rrdb':
        # Default network config used.
        torch_model = RRDBNet(in_nc = 15, out_nc = 3, nf = 64, nb = 23)
    else:
        assert False, "Conversion for model type " + args.model_type + " not implemented."

    state = torch.load(args.model_state)
    torch_model.load_state_dict(state,strict=False)
    if args.use_gpu:
        torch_model = torch_model.cuda()

    dummy_input = torch.rand([1, 15, 512, 512])
    if args.use_gpu:
        dummy_input = dummy_input.cuda()

    scripted_model = torch.jit.script(torch_model, dummy_input)
    scripted_model.save(args.output_path)
