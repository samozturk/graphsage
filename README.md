 What the code does:

1. Extracts unique entities: Gets unique diagnostic actions and questions from your DataFrame

2. Creates embeddings: Uses SentenceTransformer to create embeddings for both diagnostic titles and question texts

3. Builds the graph:

 - Diagnostic-Diagnostic edges: Based on cosine similarity of their embeddings
 - Diagnostic-Question edges: Based on your diagnostic_action_id relationships


4. Trains GraphSAGE: Learns new embeddings that incorporate both semantic content and graph structure

#### Key features:

- Handles duplicates: Automatically handles cases where same diagnostic appears multiple times
- Flexible similarity threshold: Adjust cosine_threshold to control how many diagnostic actions connect to each other
- Built-in embeddings: Uses SentenceTransformer (you can also provide pre-computed embeddings)
- Node tracking: Keeps track of which graph nodes correspond to which original IDs
<br>
The resulting embeddings will be much richer than the original sentence embeddings because they incorporate the relationship structure between diagnostics and questions!RetryClaude does not have the ability to run the code it generates yet.Claude can make mistakes. Please double-check responses.

