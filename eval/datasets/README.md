# Open-Source Issue Evaluation Map

This directory will hold 25 high-quality, verified GitHub Issues derived from open-source repositories to test SkeletonGraph against Native IDEs.

## The Targeted Codebases
We will select issues from popular repositories spanning multiple languages to prove our polyglot parser extraction and graph assembly:
1. **Python:** `pallets/flask`, `psf/requests`
2. **TypeScript:** `expressjs/express`, `facebook/react`
3. **Go:** `gin-gonic/gin`
4. **Java:** `spring-projects/spring-boot`
5. **Rust:** `tokio-rs/tokio`

## Dataset Structure (`task_XXX.json`)
Each task will be stored in a JSON file containing the ground truth.

```json
{
  "task_id": "flask_issue_4052",
  "repo": "pallets/flask",
  "commit_hash_before_fix": "a1b2c3d",
  "language": "python",
  "issue_prompt": "Routing bug: When using `add_url_rule` and the path ends with a slash, strict_slashes logic is bypassed causing a 404.",
  "expected_diff_semantics": "The `werkzeug/routing.py` map initialization must pass `strict_slashes=self.strict_slashes` explicitly on line 144.",
  "results": {
    "native_run": {
      "logs_extracted_from": ".claude.json",
      "context_dump": "...",
      "token_count": 8900
    },
    "skeletongraph_run": {
      "logs_extracted_from": "skeletongraph serve output logs",
      "context_dump": "...",
      "token_count": 1200
    }
  }
}
```

### The Live Testing Workflow (Gemini Judge)
Once you run the issue locally using `pip install -e .` and pull the logs into the JSON above, you will use `python compile_prompt.py --task flask_issue_4052`. 
This script will inject the JSON data into `eval/judge/master_prompt.txt`. 
You can then directly paste the compiled result into **Google AI Studio (Gemini 3.1 Pro)** to get the finalized benchmark TRR and SR scores.
