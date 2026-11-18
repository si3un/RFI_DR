import torch


class ModelChecker():
    def __init__(self, model):
        self.model = model
        self.get_init_params()

    def get_init_params(self):
        self.init_model = dict()
        for k, v in self.model.state_dict().items():
            # Under DDP, state_dict keys get a 'module.' prefix, so strip it
            # up front so they can be compared later in check_with_init.
            k = k[len('module.'):] if k.startswith('module.') else k
            # [Changed] Using only .cpu() without .clone(): if the tensor is already on CPU (= no GPU at all),
            #  .cpu() returns the same tensor without copying, so the "initial snapshot" actually points
            #  to the live parameter. Then, as training changes the parameter, the snapshot changes
            #  with it and everything is wrongly reported as "always freezed".
            self.init_model[k] = v.detach().clone().cpu()

    def requires_grad_table(self):
        # [Added] Return the same info as print_requires_grad as a list of (name, requires_grad) tuples,
        #  so the initial freeze state can be copied into a table in report.html as-is.
        return [(k, v.requires_grad) for k, v in self.model.named_parameters()]

    def print_requires_grad(self, verbose=True):
        for k, grad in self.requires_grad_table():
            color_print(f"{k} : {grad}", 'blue' if grad else 'red')
        if verbose:
            learnable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            non_learnable_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
            total_params = learnable_params + non_learnable_params

            print(f"Learnable params: {learnable_params:,}")
            print(f"Non-learnable params: {non_learnable_params:,}")
            print(f"Total params: {total_params:,}")
            print(f"Percentage learnable: {learnable_params / total_params * 100:.2f}%")

    def check_with_init(self, current_model, verbose=True):
        # If the model is wrapped in DDP, state_dict keys have a 'module.' prefix,
        # so strip it and compare with the same keys as the prefix-free initial snapshot.
        current_state = current_model.state_dict()
        current_state = {
            (k[len('module.'):] if k.startswith('module.') else k): v
            for k, v in current_state.items()
        }

        init_keys = set(self.init_model.keys())
        current_keys = set(current_state.keys())
        if init_keys != current_keys:
            raise ValueError(f"Key mismatch: {init_keys.symmetric_difference(current_keys)}")

        # [Changed] Also return the results as a list of (name, 'freezed'/'updated', diff).
        #  This lets report.html show "which parameters actually changed after 5 epochs" as-is
        #  (previously it was only printed to stdout and discarded).
        results = []
        for k, v0 in self.init_model.items():
            v1 = current_state[k].detach().cpu()
            if torch.equal(v0, v1):
                status, diff = 'freezed', 0.0
                if verbose:
                    color_print(f'{k} \U0001F976 freezed', 'blue')
            else:
                diff = (v0 - v1).sum().item()
                status = 'updated'
                if verbose:
                    color_print(f'{k} \U0001F525 updated', 'red', end=f' difference : {diff}\n')
            results.append((k, status, diff))
        return results


def color_print(text, color='red', bg_color=None, bold=False, underline=False, end=None):
    # make all types to string.
    text = f'{text}'
    color_dict = {
        # Basic ANSI 8 colours
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',

        # Bright variants (bright, light)
        'light_black': '\033[90m',
        'light_red': '\033[91m',
        'light_green': '\033[92m',
        'light_yellow': '\033[93m',
        'light_blue': '\033[94m',
        'light_magenta': '\033[95m',
        'light_cyan': '\033[96m',
        'light_white': '\033[97m',

        # Extra colour names (no duplicates)
        'gray': '\033[90m',
        'grey': '\033[90m',
        'orange': '\033[33m',   # mapped to yellow
        'pink': '\033[95m',     # bright magenta
        'purple': '\033[35m',   # mapped to magenta
        'lime': '\033[92m',     # light green
        'teal': '\033[36m',     # cyan
        'navy': '\033[34m',     # blue
        'maroon': '\033[31m',   # dark red
        'olive': '\033[33m'     # ochre
    }

    bg_color_dict = {
        'black': '\033[40m',
        'red': '\033[41m',
        'green': '\033[42m',
        'yellow': '\033[43m',
        'blue': '\033[44m',
        'magenta': '\033[45m',
        'cyan': '\033[46m',
        'white': '\033[47m',

        'light_black': '\033[100m',
        'light_red': '\033[101m',
        'light_green': '\033[102m',
        'light_yellow': '\033[103m',
        'light_blue': '\033[104m',
        'light_magenta': '\033[105m',
        'light_cyan': '\033[106m',
        'light_white': '\033[107m',

        # Extra colours
        'gray': '\033[100m',
        'grey': '\033[100m',
        'orange': '\033[43m',
        'pink': '\033[105m',
        'purple': '\033[45m',
        'lime': '\033[102m',
        'teal': '\033[46m',
        'navy': '\033[44m',
        'maroon': '\033[41m',
        'olive': '\033[43m'
    }

    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'

    if bold:
        text = BOLD + text
    if underline:
        text = UNDERLINE + text
    if color:
        text = color_dict[color.lower()] + text
    if bg_color:
        text = bg_color_dict[bg_color.lower()] + text
    print(text + ENDC, end=end)
