from rfdetr import RFDETRMedium


def main():
    # Load a model
    model = RFDETRMedium()

    # Train the model
    model.train(
        dataset_dir="../data/markush_annotations",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="ckpt/markush_RFDETRMedium_0618",
    )


if __name__ == "__main__":
    main()
