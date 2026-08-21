from raglean.compression.context_compressor import ContextCompressor


def test_short_context_is_left_untouched():
    compressor = ContextCompressor(token_budget=1000)
    chunks = ["The refund window is 30 days."]
    result = compressor.compress("What is the refund window?", chunks)
    assert result.tokens_after == result.tokens_before


def test_long_context_gets_trimmed_under_budget():
    compressor = ContextCompressor(token_budget=30)
    chunks = [
        "The refund window is 30 days for physical products. " * 5,
        "Shipping fees are non-refundable unless the item is damaged. " * 5,
        "Unrelated filler sentence about company history and founding year. " * 5,
    ]
    result = compressor.compress("What is the refund window?", chunks)
    assert result.tokens_after < result.tokens_before
    assert result.tokens_after <= 45  # some slack over budget from the last sentence added


def test_relevant_sentence_is_kept_over_irrelevant_ones():
    compressor = ContextCompressor(token_budget=15)
    chunks = [
        "The refund window is 30 days for physical products.",
        "Our office was founded in 2015 in a small garage downtown.",
        "The company mascot is a golden retriever named Max.",
    ]
    result = compressor.compress("What is the refund window?", chunks)
    assert "refund" in result.compressed_text.lower()
