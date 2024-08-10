import Levenshtein
from tqdm import tqdm


def calculate_similarity(seq1, seq2):
    """
    计算两个序列之间的相似度，使用Levenshtein距离计算相似度比例。
    """
    distance = Levenshtein.distance(seq1, seq2)
    max_len = max(len(seq1), len(seq2))
    similarity = 1 - distance / max_len
    return similarity


def find_most_similar_sequence(generated_sequence, family_sequences):
    """
    在家族序列中查找与生成序列相似度最高的序列，并返回相似度。
    使用tqdm显示进度条。
    """
    best_similarity = 0
    best_sequence = None

    for family_seq in tqdm(family_sequences, desc=f"Comparing with {generated_sequence}"):
        similarity = calculate_similarity(generated_sequence, family_seq.strip())
        if similarity > best_similarity:
            best_similarity = similarity
            best_sequence = family_seq.strip()

    return best_sequence, best_similarity


def load_family_sequences(file_path):
    """
    从文本文件中加载家族序列，每行一个蛋白质序列。
    """
    with open(file_path, 'r') as file:
        family_sequences = file.readlines()
    return family_sequences


def main():
    # 指定生成的蛋白质序列列表
    generated_sequences = [
'MKVTGAGGLCILKNSTEQVLLGVSTRKGLALLGTFDKKSTEGLKEAEGDEELFPEGGTEESVSEKGEETYLRYKVPEEETVEVGVLELKVRGRKEKMSL',

'MKVTGAGGLCILKNFMELQAGRHQEDETLSPQVPIRQFAARPFFLKGLQTTINGGSDGEVQGRGKGEEHPHERAEAADKGSSSHKLVKEEMREHGVARV',

'MKVTGAGGLCILKNFDPSDGQVIEMEQVPILVAGPIAIITKDTLATKAGQKGVLFAATQMPIPSEGTQEELTLELSEDAIEEMRALLLRRLRGYGKTLV',

'MKVTGAGGLCILKNAPIKLVKVLPRLKDDGHKSEGIEAVSEEPELIQINIAPILAQTKEAKPSEKKIVLSVGGGQSDGGTTVFFKQFPLARVTVDSISK'

    ]

    # 指定家族序列文件路径
    family_sequence_file = "train_data.txt"

    # 加载家族序列
    family_sequences = load_family_sequences(family_sequence_file)

    # 对每一个生成的序列，找到最相似的家族序列及其相似度
    for generated_sequence in generated_sequences:
        best_sequence, best_similarity = find_most_similar_sequence(generated_sequence, family_sequences)
        print(f"\n生成序列: {generated_sequence}")
        print(f"最相似的家族序列: {best_sequence}")
        print(f"相似度: {best_similarity:.2f}\n")


if __name__ == "__main__":
    main()
