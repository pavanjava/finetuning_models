import pandas as pd
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split


class FinetuningDatasetCreator:
    def __init__(self):
        pass

    def create_hugging_face_dataset(self, df, test_size=0.2, val_size=0.1, random_state=42):
        train_val_set, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=None)
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val_set,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=None
        )
        # convert to hf dataset from pandas dataset
        train_dataset = Dataset.from_pandas(train.reset_index(drop=True))
        val_dataset = Dataset.from_pandas(val.reset_index(drop=True))
        test_dataset = Dataset.from_pandas(test.reset_index(drop=True))

        # Create DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })

        print(f"Dataset split summary:")
        print(f"Total samples: {len(df)}")
        print(f"Train: {len(train_dataset)} samples ({len(train_dataset)/len(df)*100:.1f}%)")
        print(f"Validation: {len(val_dataset)} samples ({len(val_dataset)/len(df)*100:.1f}%)")
        print(f"Test: {len(test_dataset)} samples ({len(test_dataset)/len(df)*100:.1f}%)")

        return dataset_dict

    def push_to_hub(self, dataset_dict, repo_name, token=None, private=False):
        try:
            dataset_dict.push_to_hub(
                repo_id=repo_name,
                token=token,
                private=private
            )
            print(f"Dataset successfully pushed to: https://huggingface.co/datasets/{repo_name}")
        except Exception as e:
            print(f"Error pushing to hub: {e}")
            print("Make sure you're logged in with: huggingface-cli login")

if __name__ == '__main__':
    df = pd.read_csv('ft_pubmedqa.csv')
    ft_dataset_creator = FinetuningDatasetCreator()
    dataset_dict = ft_dataset_creator.create_hugging_face_dataset(df)

    # Push to Hugging Face Hub (recommended)
    print("\nPushing to Hugging Face Hub...")
    ft_dataset_creator.push_to_hub(dataset_dict, repo_name="pavanmantha/pubmedqa-dataset", token ="hf_", private=False)