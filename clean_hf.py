from argparse import ArgumentParser
import signal
import sys

from huggingface_hub import HfApi

# Initialize the API
api = HfApi()

def signal_handler(sig, frame):
    print('\nOperation cancelled by user. Exiting safely...')
    sys.exit(0)

def clean_private(artifact_type):
    # Register the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Fetch all datasets for the user 'ankner'
    if artifact_type == "dataset":
        artifacts = api.list_datasets(author='ankner')
    elif artifact_type == "model":
        artifacts = api.list_models(author='ankner')
    else:
        raise ValueError(f"Invalid artifact type: {artifact_type}")

    # Iterate over the datasets and check if they are private
    for artifact in artifacts:
        if artifact.private:
            try:
                # Ask the user if they want to delete the dataset
                print("="*100)
                response = input(f"Do you want to delete the private {artifact_type} '{artifact.id}'? (yes/no): ").strip().lower()
                if response == 'yes':
                    # Delete the dataset
                    api.delete_repo(repo_id=artifact.id, repo_type=artifact_type)
                    print(f"Deleted {artifact_type} '{artifact.id}'.")
                else:
                    print(f"Skipped {artifact_type} '{artifact.id}'.")
                print("="*100)
            except KeyboardInterrupt:
                signal_handler(signal.SIGINT, None)
    print("\n\nDone!")

def inspect_non_collection(artifact_type):
    # Fetch all datasets for the user 'ankner'
    if artifact_type == "dataset":
        artifacts = api.list_datasets(author='ankner')
    elif artifact_type == "model":
        artifacts = api.list_models(author='ankner')
    else:
        raise ValueError(f"Invalid artifact type: {artifact_type}")

    # Fetch all collections for the user 'ankner'
    collections = api.list_collections(owner='ankner')

    # Create a set of artifact IDs that are part of collections
    collection_artifacts = set()
    for collection in collections:
        full_collection = api.get_collection(collection.slug)
        for collection_item in full_collection.items:
            if collection_item.item_type == artifact_type:
                collection_artifacts.add(collection_item.item_id)

    # Iterate over the datasets and check if they are not in any collection
    for artifact in artifacts:
        if artifact.id not in collection_artifacts:
            print(f"{artifact_type} '{artifact.id}' is not in any collection.")

def clean_collection(collection_slug):
    collection = api.get_collection(collection_slug)
    for item in collection.items:
        try:
            # Ask user if they want to delete this item
            print("="*100)
            response = input(f"Do you want to delete {item.item_type} '{item.item_id}' from collection? (yes/no): ").strip().lower()
            if response == 'yes':
                api.delete_repo(repo_id=item.item_id, repo_type=item.item_type)
                print(f"Deleted {item.item_type} '{item.item_id}'.")
            else:
                print(f"Skipped {item.item_type} '{item.item_id}'.")
            print("="*100)
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cleanup-mode", type=str, choices=["private", "non-collection", "collection"])
    parser.add_argument("--artifact-type", type=str, choices=["dataset", "model"])
    parser.add_argument("--collection-slug", type=str, default=None)
    args = parser.parse_args()

    if args.cleanup_mode == "private":
        clean_private(args.artifact_type)
    elif args.cleanup_mode == "non-collection":
        inspect_non_collection(args.artifact_type)
    elif args.cleanup_mode == "collection":
        clean_collection(args.collection_slug)
