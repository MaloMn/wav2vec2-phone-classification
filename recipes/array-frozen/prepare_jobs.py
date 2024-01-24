import os
import sys


TEMP_DIR = "recipes/array-frozen/temp/"



def process_binary(filename, destination, layer_id):
    """ Replace strings using binary and string replace
        Processing follows original code flow except using
        binary files and string replace """

    # Map using binary strings
    replace_tokens = {b'${{LAYER_ID}}': str(layer_id).encode()}

    print(destination)

    with open(filename, 'rb') as fi, open(destination, 'wb') as fo:
        for line in fi:
            for token in replace_tokens:
                line = line.replace(token, replace_tokens[token])
            fo.write(line)



if __name__ == "__main__":
    layer_id = int(sys.argv[1])
    save_path = f"{TEMP_DIR}{layer_id}/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    process_binary("recipes/array-frozen/wav2vec2_phoneme.yml", save_path + "temp.yml", layer_id)
