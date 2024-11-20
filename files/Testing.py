import re
from pprint import pprint
import evaluate
import nltk
import numpy as np
import torch
from datasets import load_dataset
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer, \
    T5ForConditionalGeneration, AutoModelWithLMHead


class QuestionGenerator:
    def __init__(self, model_type="squad"):
        self.model_type = model_type
        self.metric = evaluate.load("rouge")
        if model_type == "squad":
            self.tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer",
                                                           cache_dir='D:/cache/', device_map="auto")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer",
                                                               cache_dir='D:/cache/', device_map="auto")
        elif model_type == "race":
            self.tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-race-QuestionAnswer",
                                                           cache_dir='D:/cache/', device_map="auto", )
            self.model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-race-QuestionAnswer",
                                                               cache_dir='D:/cache/', device_map="auto")
        elif model_type == "small":
            self.tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap",
                                                           cache_dir='D:/cache/', device_map="auto")
            self.model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap",
                                                             cache_dir='D:/cache/', device_map="auto")
        else:
            raise ValueError("Non-implemented model parameter")
        self.answer_gen_model = T5ForConditionalGeneration.from_pretrained("./t5_ans_gen_model/t5_base_ans_gen",
                                                                           cache_dir="D:/cache", device_map="cuda")
        self.answer_gen_tokenizer = T5Tokenizer.from_pretrained("./t5_ans_gen_model/t5_base_tok_ans_gen",
                                                                cache_dir="D:/cache",
                                                                device_map="cuda")
        self.distractor_model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-race-Distractor")
        self.distractor_tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-race-Distractor")

    def needs_answers(self):
        if self.model_type == 'race' or self.model_type == "squad":
            return False
        else:
            return True

    def get_contexts(self, text, contexts_model_type="naive"):
        if contexts_model_type == 'naive':
            sentences = nltk.tokenize.sent_tokenize(text)

            # get steps of 3 sentences
            prev = None
            for i, sentence in enumerate(sentences):
                if i + 1 != len(sentences):
                    after = sentences[i + 1]
                else:
                    after = None

                if prev is not None and after is not None:
                    yield f"{prev} {sentence} {after}"

                prev = sentence


        else:
            # todo: implement better, less naive context generating models
            raise NotImplementedError

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        print(logits[1].shape)
        predictions = np.argmax(logits[0], axis=-1)
        predictions = [self.tokenizer.decode(prediction) for prediction in predictions]
        labels = [self.tokenizer.decode(label) for label in labels]

        return self.metric.compute(predictions=predictions, references=labels)

    def tokenize_function(self, examples):
        model_inputs = self.tokenizer(examples["sentence"], padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(examples["labels"], padding="max_length", truncation=True, return_tensors="pt")
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def few_shot_train(self):
        if self.model_type != "small":
            data = load_dataset(path="./data_squad")
        else:
            data = load_dataset(path="./data_base")
        train_data = data["train"].map(self.tokenize_function, batched=True)
        test_data = data["test"].map(self.tokenize_function, batched=True)
        if self.model_type == "small":
            epochs = 5
            learning_rate = 0.00002
        else:
            epochs = 10
            learning_rate = 0.0001
        training_arguments = Seq2SeqTrainingArguments(output_dir="test_trainer",
                                                      evaluation_strategy="epoch",
                                                      per_device_train_batch_size=1,
                                                      per_device_eval_batch_size=1,
                                                      num_train_epochs=epochs,
                                                      gradient_accumulation_steps=5,
                                                      gradient_checkpointing=True,
                                                      bf16=True,
                                                      learning_rate=learning_rate)
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_data,
            eval_dataset=test_data,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

    # General script to generate questions based on a .txt or .pdf file. .txt files work better as of now, but that can
    # easily be fixed with a little tweaking
    def get_distractors(self, question, answer, context):
        input_text = " ".join(
            [question, self.distractor_tokenizer.sep_token, answer, self.distractor_tokenizer.sep_token, context])
        inputs = self.distractor_tokenizer(input_text, return_tensors="pt")
        outputs = self.distractor_model.generate(**inputs, max_new_tokens=128)
        distractors = self.distractor_tokenizer.decode(outputs[0], skip_special_tokens=False)
        distractors = distractors.replace(self.tokenizer.pad_token, "").replace(self.tokenizer.eos_token, "")
        distractors = [y.strip() for y in distractors.split(self.tokenizer.sep_token)]
        return distractors

    def get_answer(self, context):
        sents = context.split(". ")
        if len(sents) == 3:
            input_text = f"{sents[0]}. [HL] {sents[1]}. [HL] {sents[2]}."
            input_text = input_text.strip()
            inputs = self.answer_gen_tokenizer(input_text, max_length=512, pad_to_max_length=True,
                                               return_tensors='pt').to(
                "cuda")
            output = self.answer_gen_model.generate(**inputs, max_length=100)
            answer = self.answer_gen_tokenizer.decode(output[0], skip_special_tokens=True)
            print(answer.replace("[SEP]", "").strip())
            return answer.replace("[SEP]", "").strip()
        else:
            return None

    def generate(self, file, train=True):
        text = ''

        # Checks the filetype and extracts text. Only works for .txt and .pdf files.
        if file[-4:] == ".pdf":
            pdfreader = PdfReader(file)
            for page in pdfreader.pages:
                text += page.extract_text()
        elif file[-4:] == ".txt":
            with open(file, 'r', encoding="utf-8") as f:
                text = f.read()
        else:
            print("Invalid file type.")
            return

        # Prints the text after some special character removal.
        text = re.sub("[^A-Za-z0-9 -_.,!?\"'/$# ()]", "", text)
        pprint(text)

        # few shot train
        if train:
            self.few_shot_train()
            self.model = self.model.to("cuda")

        contexts = self.get_contexts(text)
        output = []
        i = 0
        for context in contexts:  # remove indexing later, pair with testing of qnauestions and answers
            answer = ""
            if i >= 10:
                break
            i += 1
            if self.needs_answers():
                answer = self.get_answer(context)
                if answer is None:
                    continue
                print(answer)
                context = f"answer: {answer}  context: {context} </s>"
            inputs = self.tokenizer(context, return_tensors="pt", return_overflowing_tokens=False).to("cuda")
            outputs = self.model.generate(**inputs, max_length=100)
            print(outputs[0])

            if self.model_type != "small":
                question_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                question_answer = question_answer.replace(self.tokenizer.pad_token, "")\
                    .replace(self.tokenizer.eos_token, "")
                question, answer = question_answer.split(self.tokenizer.sep_token)
                print(question_answer)
            else:
                question_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                question_answer = question_answer.replace(self.tokenizer.pad_token, "")\
                    .replace(self.tokenizer.eos_token, "")
                question = question_answer.replace("question:", "").strip()

            distractors = self.get_distractors(question, answer, context)
            output.append(f"Question: {question} Answer: {answer} Alt. Answers: {distractors}")

        filename = file[:-4] + "_output.txt"
        if train:
            filename = filename[:-4] + "_trained.txt"
        if self.model_type == "small":
            filename = filename[:-4] + "_small.txt"

        with open(filename, 'w', encoding="utf-8") as f:
            f.write(str(output))

        torch.cuda.empty_cache()

        # Print output to console
        # pprint(output)


small_question_generator = QuestionGenerator("small")
small_question_generator.generate("Pilot_4.txt", train=False)
small_question_generator.generate("Pilot_4.txt")
del small_question_generator

squad_qgen = QuestionGenerator()
squad_qgen.generate("Pilot_2.txt", False)
squad_qgen.generate("Pilot_2.txt", True)
del squad_qgen

race_qgen = QuestionGenerator("race")
race_qgen.generate("Pilot_1.txt", False)
race_qgen.generate("Pilot_2.txt", True)