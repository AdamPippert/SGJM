from sgjm.graph.address import AddressBook, Signature


def test_signature_from_tokens_is_deterministic():
    a = Signature.from_tokens([1, 2, 3])
    b = Signature.from_tokens([1, 2, 3])
    assert a == b
    assert a.hamming(b) == 0


def test_signature_from_latent_close_for_similar_vectors():
    base = [0.1 * i for i in range(16)]
    perturbed = [v + 1e-3 for v in base]
    s1 = Signature.from_latent(base)
    s2 = Signature.from_latent(perturbed)
    assert s1.hamming(s2) <= 4


def test_address_book_resolves_repeated_signatures_to_same_address():
    book = AddressBook(merge_radius=0)
    sig = Signature.from_tokens([7, 8, 9])
    a, fresh = book.resolve_or_allocate(sig)
    assert fresh
    b, fresh2 = book.resolve_or_allocate(sig)
    assert not fresh2
    assert a == b


def test_address_book_merges_within_radius():
    book = AddressBook(merge_radius=8)
    base = [0.1 * i for i in range(16)]
    perturbed = [v + 1e-3 for v in base]
    a, _ = book.resolve_or_allocate(Signature.from_latent(base))
    b, fresh = book.resolve_or_allocate(Signature.from_latent(perturbed))
    assert not fresh
    assert a == b
