from corpus import MAX_WORDS, chunk_document, load_corpus, split_paragraphs, strip_frontmatter
from multi_dir_corpus import find_md_leaf_dirs, load_corpus_recursive


def words(n):
    return " ".join(["word"] * n)


def test_frontmatter_is_stripped():
    assert strip_frontmatter("---\ntitle: x\n---\nbody").strip() == "body"
    assert strip_frontmatter("no frontmatter") == "no frontmatter"


def test_fenced_code_block_stays_in_one_paragraph():
    text = "intro\n\n```\nline one\n\nline two\n```\n\noutro"
    paragraphs = split_paragraphs(text)
    assert any("line one" in p and "line two" in p for p in paragraphs)


def test_chunks_respect_the_word_ceiling_and_are_numbered():
    text = "\n\n".join(words(150) for _ in range(10))
    chunks = chunk_document("doc.md", text)
    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.word_count <= MAX_WORDS for c in chunks)
    assert sum(c.word_count for c in chunks) == 1500


def test_sample_corpus_loads(tmp_path):
    (tmp_path / "a.md").write_text(words(250))
    (tmp_path / "b.md").write_text(words(50))
    chunks = load_corpus([tmp_path])
    assert {c.source_file.rsplit("/", 1)[-1] for c in chunks} == {"a.md", "b.md"}


def test_recursive_loader_finds_nested_dirs_and_subsamples_reproducibly(tmp_path):
    for sub in ("one", "two/deep"):
        d = tmp_path / sub
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"{i}.md").write_text(words(300))
    assert len(find_md_leaf_dirs(tmp_path)) == 2
    first, _ = load_corpus_recursive(tmp_path, max_chunks=3, seed=7)
    second, _ = load_corpus_recursive(tmp_path, max_chunks=3, seed=7)
    assert len(first) == 3
    assert [c.source_file for c in first] == [c.source_file for c in second]
