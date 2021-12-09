from depth_masking_validate import DepthMaskingValidate
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    validate = DepthMaskingValidate(opts)
    validate.run_validation()