""" 1. Templates for portrait """
portrait_temp_pos = "ultra-detailed, raw photo, " \
                    "a photo of {}, single person, 50mm" \
                    "looking at viewer, " \
                    "cinematic light, perfect eyes, perfect face, soft light, " \
                    "best illustration, best shadow, masterpiece, best quality, " \
                    "extremely detailed eyes and face, detailed nose, perfect face, " \
                    "realistic, ultra-high res, " \
                    "detailed fingers, " \
                    "realistic body, "

portrait_temp_neg = "blurry face, overexpose, multiple persons, highlight on face, high contrast ratio, " \
                    "only eyes, extra fingers, fewer fingers, " \
                    "grayscale, monochrome, paintings, normal quality, " \
                    "skin spots, acnes, skin blemishes, age spot, glans, " \
                    "bad hands, missing fingers, missing arms, extra arms, malformed limbs, " \
                    "fused fingers, too many fingers, mutated hands, multi nipples, " \
                    "missing legs, extra legs, extra digit, fewer digits," \
                    "bad anatomy, text, error, cross-eyed, polar lowres" \
                    "blurry, poorly drawn hands, poorly drawn face, mutation, deformed" \
                    "EasyNegative, bad proportion body to legs, " \
                    "big head, wrong toes, extra toes, missing toes, weird toes" \
                    "aged up, old,"


""" 2. Templates for style """
style_temp_pos = "{}"

style_temp_neg = "blurry face, overexpose, highlight on face, " \
                 "only eyes, extra fingers, fewer fingers, divider line, "


""" 3. Templates for action single """
action_single_temp_pos = "ultra-detailed, realistic single person portrait, full frame raw, " \
                         "{}, Thin lips, no beard, Chinese man, young, with hands, detailed fingers, perfect eyes, " \
                         "soft light, realistic body, " \
                         "cinematic light, best illustration, best shadow, " \
                         "perfect face, detailed legs, masterpiece, best quality, " \
                         "extremely detailed eyes and face, detailed nose, " \
                         "ultra-high res, "

action_single_temp_neg = "only object, only person, close-up, no face, only head, no hand, no upper body, " \
                         "blurry face, with beard, old, " \
                         "overexpose, naked, multiple persons, " \
                         "paintings, highlight on face, high contrast ratio, " \
                         "only eyes, extra fingers, fewer fingers, " \
                         "grayscale, monochrome, normal quality, " \
                         "skin spots, acnes, skin blemishes, age spot, glans, " \
                         "bad hands, missing fingers, missing arms, extra arms, malformed limbs, " \
                         "fused fingers, too many fingers, mutated hands, multi nipples, " \
                         "missing legs, extra legs, extra digit, fewer digits," \
                         "bad anatomy, text, error, cross-eyed, polar lowres" \
                         "blurry, poorly drawn hands, poorly drawn face, mutation, deformed" \
                         "EasyNegative, bad proportion body to legs, " \
                         "big head, wrong toes, extra toes, missing toes, weird toes"


""" 4. Templates for action two """
action_two_temp_pos = "ultra-detailed, 4k res, full frame raw, {}, realistic portrait of two persons, upper bodies, " \
                      "extremely detailed eyes and faces, " \
                      "perfect faces, perfect eyes, clear eyes, with hands, " \
                      "realistic bodies, less beard, " \
                      "detailed nose, " \
                      "cinematic light, soft light, " \
                      "both looking at viewer, " \
                      "best illustration, best shadow, masterpiece, best quality, " \
                      "realistic, " \
                      "detailed fingers, "

action_two_temp_neg = "blurry face, blurry eyes, mixed faces, close-up, only object, only background, no hand, " \
                      "overexpose, fat face, " \
                      "no person, single person, no upper body, only face, only eyes, " \
                      "no face, only head, extra fingers, fewer fingers, " \
                      "highlight on face, high contrast ratio, " \
                      "grayscale, monochrome, paintings, normal quality, " \
                      "skin spots, acnes, skin blemishes, age spot, glans, " \
                      "bad hands, missing fingers, missing arms, extra arms, malformed limbs, " \
                      "fused fingers, too many fingers, mutated hands, multi nipples, " \
                      "missing legs, extra legs, extra digit, fewer digits," \
                      "bad anatomy, text, error, cross-eyed, polar lowres" \
                      "blurry, poorly drawn hands, poorly drawn face, mutation, deformed" \
                      "EasyNegative, bad proportion body to legs, " \
                      "big head, wrong toes, extra toes, missing toes, weird toes" \
                      "aged up, old,"


def get_pos_neg_temps(from_file_name: str):
    portrait_files = ['pot', ]
    style_files = ['style', 'example', ]
    action_single_files = ['single', 'tmp', ]
    action_two_files = ['two', 'celeb', ]
    if any(suffix in from_file_name for suffix in portrait_files):
        temp_pos = portrait_temp_pos
        temp_neg = portrait_temp_neg
    elif any(suffix in from_file_name for suffix in style_files):
        temp_pos = style_temp_pos
        temp_neg = style_temp_neg
    elif any(suffix in from_file_name for suffix in action_single_files):
        temp_pos = action_single_temp_pos
        temp_neg = action_single_temp_neg
    elif any(suffix in from_file_name for suffix in action_two_files):
        temp_pos = action_two_temp_pos
        temp_neg = action_two_temp_neg
    else:
        raise ValueError('Not supported from_file_name.')
    return temp_pos, temp_neg
