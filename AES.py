import numpy as np

S_BOX = [
    [99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118],
    [202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164, 114, 192],
    [183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49, 21],
    [4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117],
    [9, 131, 44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 41, 227, 47, 132],
    [83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207],
    [208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168],
    [81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 33, 16, 255, 243, 210],
    [205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115],
    [96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219],
    [224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98, 145, 149, 228, 121],
    [231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8],
    [186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138],
    [112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158],
    [225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223],
    [140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22]
]

INV_S_BOX = [
    [82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 215, 251],
    [124, 227, 57, 130, 155, 47, 255, 135, 52, 142, 67, 68, 196, 222, 233, 203],
    [84, 123, 148, 50, 166, 194, 35, 61, 238, 76, 149, 11, 66, 250, 195, 78],
    [8, 46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37],
    [114, 248, 246, 100, 134, 104, 152, 22, 212, 164, 92, 204, 93, 101, 182, 146],
    [108, 112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87, 167, 141, 157, 132],
    [144, 216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 5, 184, 179, 69, 6],
    [208, 44, 30, 143, 202, 63, 15, 2, 193, 175, 189, 3, 1, 19, 138, 107],
    [58, 145, 17, 65, 79, 103, 220, 234, 151, 242, 207, 206, 240, 180, 230, 115],
    [150, 172, 116, 34, 231, 173, 53, 133, 226, 249, 55, 232, 28, 117, 223, 110],
    [71, 241, 26, 113, 29, 41, 197, 137, 111, 183, 98, 14, 170, 24, 190, 27],
    [252, 86, 62, 75, 198, 210, 121, 32, 154, 219, 192, 254, 120, 205, 90, 244],
    [31, 221, 168, 51, 136, 7, 199, 49, 177, 18, 16, 89, 39, 128, 236, 95],
    [96, 81, 127, 169, 25, 181, 74, 13, 45, 229, 122, 159, 147, 201, 156, 239],
    [160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97],
    [23, 43, 4, 126, 186, 119, 214, 38, 225, 105, 20, 99, 85, 33, 12, 125]
]

R_CON = [1, 2, 4, 8, 16, 32, 64, 128, 27, 54]

MIX_COLUMNS_MATRIX = np.array([
    [2, 3, 1, 1],
    [1, 2, 3, 1],
    [1, 1, 2, 3],
    [3, 1, 1, 2]
])

def key_expansion(key):
    Nk = 4  
    Nr = 10  
    Nb = 4  

    def sub_word(word):
        return [S_BOX[b >> 4][b & 0x0F] for b in word]

    def rot_word(word):
        return word[1:] + word[:1]

    key_schedule = [key[i:i + 4] for i in range(0, len(key), 4)]

    for i in range(Nk, Nb * (Nr + 1)):
        temp = key_schedule[i - 1]  
        if i % Nk == 0:
            temp = sub_word(rot_word(temp))
            temp[0] ^= R_CON[(i // Nk) - 1]
        new_word = [key_schedule[i - Nk][j] ^ temp[j] for j in range(4)]
        key_schedule.append(new_word)

    round_keys = [key_schedule[i:i + Nb] for i in range(0, len(key_schedule), Nb)]
    return round_keys

def gmul(a, b):
    result = 0
    for _ in range(8):
        if b & 1:
            result ^= a
        carry = a & 0x80
        a = (a << 1) & 0xFF
        if carry:
            a ^= 0x1B
        b >>= 1
    return result

def mix_columns(state):
    for col in range(4):
        column = state[:, col]
        state[:, col] = [
            gmul(column[0], 2) ^ gmul(column[1], 3) ^ column[2] ^ column[3],
            column[0] ^ gmul(column[1], 2) ^ gmul(column[2], 3) ^ column[3],
            column[0] ^ column[1] ^ gmul(column[2], 2) ^ gmul(column[3], 3),
            gmul(column[0], 3) ^ column[1] ^ column[2] ^ gmul(column[3], 2)
        ]
    return state

def sub_bytes(state):
    return np.array([[S_BOX[byte >> 4][byte & 0x0F] for byte in row] for row in state])

def shift_rows(state):
    state[1] = np.roll(state[1], -1)
    state[2] = np.roll(state[2], -2)
    state[3] = np.roll(state[3], -3)
    return state

def add_round_key(state, round_key):
    return np.bitwise_xor(state, np.array(round_key))


def inv_sub_bytes(state):
    return np.array([[INV_S_BOX[byte >> 4][byte & 0x0F] for byte in row] for row in state])

def inv_shift_rows(state):
    state[1] = np.roll(state[1], 1)
    state[2] = np.roll(state[2], 2)
    state[3] = np.roll(state[3], 3)
    return state

def inv_mix_columns(state):
    for col in range(4):
        column = state[:, col]
        state[:, col] = [
            gmul(column[0], 14) ^ gmul(column[1], 11) ^ gmul(column[2], 13) ^ gmul(column[3], 9),
            gmul(column[0], 9) ^ gmul(column[1], 14) ^ gmul(column[2], 11) ^ gmul(column[3], 13),
            gmul(column[0], 13) ^ gmul(column[1], 9) ^ gmul(column[2], 14) ^ gmul(column[3], 11),
            gmul(column[0], 11) ^ gmul(column[1], 13) ^ gmul(column[2], 9) ^ gmul(column[3], 14)
        ]
    return state

def pad(data, block_size=16):
    padding_len = block_size - (len(data) % block_size)
    return data + [padding_len] * padding_len

def unpad(data):
    padding_len = data[-1]
    return data[:-padding_len]

def aes_encrypt(plaintext, key):
    plaintext_to_bytes = list(plaintext.encode('utf-8'))
    padded_plaintext = pad(plaintext_to_bytes)

    key_to_bytes = list(key.encode('utf-8'))
    if len(key_to_bytes) < 16:
        key_to_bytes += [0] * (16 - len(key_to_bytes))

    encrypted = []
    for i in range(0, len(padded_plaintext), 16):
        block = padded_plaintext[i:i + 16]
        encrypted.extend(aes_encrypt_block(block, key_to_bytes))

    return encrypted

def aes_decrypt(ciphertext, key):
    key_to_bytes = list(key.encode('utf-8'))
    if len(key_to_bytes) < 16:
        key_to_bytes += [0] * (16 - len(key_to_bytes))

    decrypted = []
    for i in range(0, len(ciphertext), 16):
        block = ciphertext[i:i + 16]
        decrypted.extend(aes_decrypt_block(block, key_to_bytes))

    return bytes(unpad(decrypted)).decode('utf-8')

def aes_encrypt_block(block, key):
    state = np.array(block).reshape(4, 4).T
    round_keys = key_expansion(key)

    state = add_round_key(state, round_keys[0])

    for round_num in range(1, 10):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, round_keys[round_num])

    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, round_keys[-1])

    return state.T.flatten().tolist()

def aes_decrypt_block(block, key):
    state = np.array(block).reshape(4, 4).T
    round_keys = key_expansion(key)

    state = add_round_key(state, round_keys[-1])

    for round_num in range(9, 0, -1):
        state = inv_shift_rows(state)
        state = inv_sub_bytes(state)
        state = add_round_key(state, round_keys[round_num])
        state = inv_mix_columns(state)

    state = inv_shift_rows(state)
    state = inv_sub_bytes(state)
    state = add_round_key(state, round_keys[0])

    return state.T.flatten().tolist()

print("Your key for encrypting: ")
key = input()

print("Your string for encrypting: ")
plaintext = input()

encrypted = aes_encrypt(plaintext, key)
print("Encrypted data:", encrypted)

decrypted_string = aes_decrypt(encrypted, key)
print("Decrypted string:", decrypted_string)