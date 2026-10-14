from faq import search_faq_docs

for q in ["how much does an ICU ambulance cost",
          "do I pay online",
          "what is the capital of France",
          "how much will cancelling cost"
          ]:
    print("\n==", q)
    for d in search_faq_docs(q):
        print("  ", d.metadata["source"], "|", d.page_content.splitlines()[0])