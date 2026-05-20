from transformers import HfArgumentParser

from distill_retriever import (
    EncoderOnlyEmbedderDistillDataArguments,
    EncoderOnlyEmbedderDistillModelArguments,
    EncoderOnlyEmbedderDistillRunner,
    EncoderOnlyEmbedderDistillTrainingArguments,
)


def main():
    parser = HfArgumentParser(
        (
            EncoderOnlyEmbedderDistillModelArguments,
            EncoderOnlyEmbedderDistillDataArguments,
            EncoderOnlyEmbedderDistillTrainingArguments,
        )
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: EncoderOnlyEmbedderDistillModelArguments
    data_args: EncoderOnlyEmbedderDistillDataArguments
    training_args: EncoderOnlyEmbedderDistillTrainingArguments

    runner = EncoderOnlyEmbedderDistillRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    runner.run()


if __name__ == "__main__":
    main()
