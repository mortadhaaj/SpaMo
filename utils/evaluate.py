from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF, TER
import re


def is_arabic_text(text):
    """Check if text contains Arabic characters."""
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
    return bool(arabic_pattern.search(text))


def calculate_rouge_char_based(references, predictions):
    """
    Calculate ROUGE-L using character-based tokenization for Arabic text.
    This works better for Arabic and other non-space-delimited languages.
    """
    def lcs_length(s1, s2):
        """Calculate longest common subsequence length."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    precisions, recalls, f1s = [], [], []

    for ref, pred in zip(references, predictions):
        # Remove spaces and periods for character-based matching
        ref_chars = ref.replace(' ', '').replace('.', '')
        pred_chars = pred.replace(' ', '').replace('.', '')

        if len(ref_chars) == 0 or len(pred_chars) == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue

        lcs_len = lcs_length(ref_chars, pred_chars)

        precision = lcs_len / len(pred_chars) if len(pred_chars) > 0 else 0.0
        recall = lcs_len / len(ref_chars) if len(ref_chars) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        'precision': sum(precisions) / len(precisions) if precisions else 0.0,
        'recall': sum(recalls) / len(recalls) if recalls else 0.0,
        'f1': sum(f1s) / len(f1s) if f1s else 0.0
    }


def evaluate_results(predictions, references, split="train", device='cpu', tokenizer='13a'):
    """
    Evaluate prediction results using BLEU and ROUGE metrics.

    Args:
        predictions (list): List of predicted sequences.
        references (list): List of reference sequences.
        tokenizer (object, optional): Tokenizer if needed for evaluation.
        split (str): The data split being evaluated.

    Returns:
        dict: A dictionary of evaluation scores.
    """
    log_dicts = {}

    bleu4 = BLEU(max_ngram_order=4, tokenize=tokenizer).corpus_score(predictions, [references]).score
    log_dicts[f"{split}/bleu4"] = bleu4

    if split == 'test':
        for i in range(1, 4):
            score = BLEU(max_ngram_order=i, tokenize=tokenizer).corpus_score(predictions, [references]).score
            log_dicts[f"{split}/bleu" + str(i)] = score

        # Calculate ROUGE-L score
        # Check if text is Arabic and use appropriate method
        if references and is_arabic_text(references[0]):
            # Use character-based ROUGE for Arabic text
            rouge_results = calculate_rouge_char_based(references, predictions)
            log_dicts[f"{split}/rougeL_precision"] = rouge_results['precision'] * 100
            log_dicts[f"{split}/rougeL_recall"] = rouge_results['recall'] * 100
            log_dicts[f"{split}/rougeL_f1"] = rouge_results['f1'] * 100
        else:
            # Use standard ROUGE for English and other Latin-script languages
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_scores = [scorer.score(ref, pred)['rougeL'] for ref, pred in zip(references, predictions)]

            # Aggregate ROUGE-L scores (average precision, recall, and F1)
            avg_precision = sum(score.precision for score in rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
            avg_recall = sum(score.recall for score in rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
            avg_f1 = sum(score.fmeasure for score in rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

            log_dicts[f"{split}/rougeL_precision"] = avg_precision * 100
            log_dicts[f"{split}/rougeL_recall"] = avg_recall * 100
            log_dicts[f"{split}/rougeL_f1"] = avg_f1 * 100

    return log_dicts