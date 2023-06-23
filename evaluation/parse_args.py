import argparse


def parser_main(parser):
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="exp_eval/ours/"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        # default="configs/stable-diffusion/v1-inference.yaml",
        default="configs/stable-diffusion/aigc_id_infer.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./weights/sd-v1-4-full-ema.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )


    parser.add_argument(
        "--embedding_path",
        type=str,
        help="Path to a pre-trained embedding manager checkpoint")

    parser.add_argument(
        "--img_suffix",
        type=str,
        default="",
        help="The suffix of saved images.")

    return parser


def parser_gen(parser):
    parser.add_argument(
        "--eval_out_dir",
        type=str,
        default="exp_eval/ours/",
        help="Output root dir.")
    parser.add_argument(
        "--eval_project_folder",
        type=str,
        help="The training project folder where weights saved here.")
    parser.add_argument(
        "--eval_step",
        type=int,
        default=19999,
        help="The step of the model weights.")
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default='vgg1',
        help="Test target dataset.")
    parser.add_argument(
        "--eval_id",
        type=str,
        default=0,
        help="The id (e.g. 0, 0-10) of the first person.")
    parser.add_argument(
        "--eval_id2",
        type=str,
        default="",
        help="The id (e.g. 0, 0-10) of the second person.")
    parser.add_argument(
        "--eval_img_idx",
        type=int,
        default=0,
        help="The image idx of the person.")
    parser.add_argument(
        "--eval_not_use_saved_id",
        action="store_false",
        help="If stored, not use saved_id.")
    parser.add_argument(
        "--eval_resume_folder",
        type=str,
        default="",
        help="Resume from which folder.")
    parser.add_argument(
        "--eval_resume_cnt",
        type=int,
        default=0,
        help="Resume from which prompt.")
    return parser


def parser_eval(parser):
    parser.add_argument(
        "--eval_out_dir",
        type=str,
        default="exp_eval/ours/",
        help="Output root dir. {out_dir}/{project_folder}/eval_{time}")
    parser.add_argument(
        "--eval_project_folder",
        type=str,
        help="The training project folder where weights saved here.")
    parser.add_argument(
        "--eval_time_folder",
        type=str,
        help="The evaluating (generating) time.")
    return parser
