import unittest

from scripts.build_dataset import entity_gene_pool, split_entities


class GroupedDatasetSplitTest(unittest.TestCase):
    def test_split_is_reproducible_and_disjoint(self) -> None:
        entities = [
            {"bgc_id": f"BGC{i}", "gene_ids": [f"g{i}:0", f"g{i}:1"]}
            for i in range(10)
        ]

        train, val = split_entities(entities, 0.2, seed=42)
        train_again, val_again = split_entities(entities, 0.2, seed=42)

        self.assertEqual(train, train_again)
        self.assertEqual(val, val_again)
        self.assertEqual(len(train), 8)
        self.assertEqual(len(val), 2)
        self.assertTrue(
            {row["bgc_id"] for row in train}.isdisjoint(
                row["bgc_id"] for row in val
            )
        )
        self.assertTrue(
            set(entity_gene_pool(train)).isdisjoint(entity_gene_pool(val))
        )

    def test_split_rejects_invalid_fraction(self) -> None:
        with self.assertRaises(ValueError):
            split_entities([{"gene_ids": []}], 1.0, seed=42)


if __name__ == "__main__":
    unittest.main()
