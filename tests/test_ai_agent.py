import unittest
from ai_agent import GmailAgent, _extract_json, heuristic_score


class DummyLLM:
    def __init__(self, response):
        self.response = response
    def generate(self, prompt, temperature=0.2):
        return self.response


class AgentTests(unittest.TestCase):
    def test_heuristic_scores_deadline_above_newsletter(self):
        urgent, _ = heuristic_score({"subject": "Action required: deadline due today", "body": "Please reply"})
        low, _ = heuristic_score({"subject": "Weekly newsletter sale ends", "body": "unsubscribe"})
        self.assertGreater(urgent, low)

    def test_extract_json_handles_fenced_response(self):
        result = _extract_json('```json\n{"daily_summary":"ok","top_emails":[],"action_items":[]}\n```')
        self.assertEqual(result["daily_summary"], "ok")

    def test_agent_maps_model_ids_to_known_emails(self):
        emails = [{"id": "a", "subject": "Deadline", "sender": "person@example.com", "body": "Due today", "snippet": "Due today", "internal_date": 2}]
        llm = DummyLLM('{"daily_summary":"One deadline","top_emails":[{"id":"a","summary":"Due today","importance_score":90,"importance_reason":"Deadline","action_items":["Reply"]}],"action_items":["Reply"]}')
        result = GmailAgent(llm).analyze(emails)
        self.assertEqual(result["top_emails"][0]["id"], "a")
        self.assertEqual(result["top_emails"][0]["importance_score"], 90)
        self.assertEqual(result["action_items"], ["Reply"])

    def test_invalid_model_json_falls_back(self):
        result = GmailAgent(DummyLLM("not json")).analyze([{"id": "a", "subject": "Test", "snippet": "Hello"}])
        self.assertEqual(len(result["top_emails"]), 1)
        self.assertIn("fallback", result["ranking_method"])

    def test_empty_inbox(self):
        result = GmailAgent(DummyLLM("")).analyze([])
        self.assertEqual(result["top_emails"], [])


if __name__ == "__main__":
    unittest.main()
