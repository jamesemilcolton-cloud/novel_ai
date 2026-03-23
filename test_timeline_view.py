import unittest

from novel_ai import parse_canon_memory, render_timeline_overview


class TimelineViewTests(unittest.TestCase):
    def test_timeline_overview_formats_story_map_in_chronological_order(self):
        content = """
========================
CHAPTER 2

[Timeline]
- Neutron event countdown begins in earnest (ACTIVE)

[Character]
- Manfrid hides worsening tremors during briefing (ACTIVE)

========================
CHAPTER 1

[Mission State — Active]
- Shield deployment window narrows before the storm front arrives (ACTIVE)

[Relationship State — Active]
- Crew blame shifts toward Talia over shield preparation delays (ACTIVE)
""".strip()

        overview = render_timeline_overview(parse_canon_memory(content))

        expected = """Chapter 1
• [Mission] Shield deployment window narrows before the storm front arrives (ACTIVE)
• [Relationship] Crew blame shifts toward Talia over shield preparation delays (ACTIVE)

Chapter 2
• [Mission] Neutron event countdown begins in earnest (ACTIVE)
• [Character] Manfrid hides worsening tremors during briefing (ACTIVE)"""
        self.assertEqual(overview, expected)
        self.assertNotIn("{'text'", overview)

    def test_timeline_overview_merges_duplicate_thread_until_resolution(self):
        content = """
========================
CHAPTER 1

[Timeline]
- Shield readiness conflict splits the bridge crew (ACTIVE)

========================
CHAPTER 2

[Relationship State — Active]
- Bridge crew stays divided over shield readiness responsibilities (ACTIVE)

========================
CHAPTER 5

[Mission State — Resolved]
- Shield readiness conflict resolved after emergency drill succeeds (RESOLVED)
""".strip()

        overview = render_timeline_overview(parse_canon_memory(content))

        self.assertIn("Chapter 1", overview)
        self.assertNotIn(
            "Chapter 2\n• [Relationship] Bridge crew stays divided over shield readiness responsibilities (ACTIVE)",
            overview,
        )
        self.assertIn(
            "Chapter 5\n✓ [Mission] Shield readiness conflict resolved after emergency drill succeeds (RESOLVED)",
            overview,
        )

    def test_timeline_overview_ignores_minor_atmosphere_details(self):
        content = """
========================
CHAPTER 3

[Location]
- Corridor lights cast a pale blue glow across the bulkheads (ACTIVE)

[World]
- Radiation alarm forces the crew to seal the observation deck (ACTIVE)
""".strip()

        overview = render_timeline_overview(parse_canon_memory(content))

        self.assertEqual(
            overview,
            "Chapter 3\n• [Conflict] Radiation alarm forces the crew to seal the observation deck (ACTIVE)",
        )


if __name__ == "__main__":
    unittest.main()
