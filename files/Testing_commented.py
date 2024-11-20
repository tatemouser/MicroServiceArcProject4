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
    # Function '__init__' definition
    def __init__(self, model_type="squad"):
        # Variable assignment
        self.model_type = model_type
        # Variable assignment
        self.metric = evaluate.load("rouge")
        # Conditional statement
        if model_type == "squad":
            # Variable assignment
            self.tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer",
                                                           cache_dir='D:/cache/', device_map="auto")
            # Variable assignment
            self.model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer",
                                                               cache_dir='D:/cache/', device_map="auto")
        # Conditional statement
        elif model_type == "race":
            # Variable assignment
            self.tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-race-QuestionAnswer",
                                                           cache_dir='D:/cache/', device_map="auto", )
            # Variable assignment
            self.model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-race-QuestionAnswer",
                                                               cache_dir='D:/cache/', device_map="auto")
        # Conditional statement
        elif model_type == "small":
            # Variable assignment
            self.tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap",
                                                           cache_dir='D:/cache/', device_map="auto")
            # Variable assignment
            self.model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap",
                                                             cache_dir='D:/cache/', device_map="auto")
        else:
            raise ValueError("Non-implemented model parameter")
        # Variable assignment
        self.answer_gen_model = T5ForConditionalGeneration.from_pretrained("./t5_ans_gen_model/t5_base_ans_gen",
                                                                           cache_dir="D:/cache", device_map="cuda")
        # Variable assignment
        self.answer_gen_tokenizer = T5Tokenizer.from_pretrained("./t5_ans_gen_model/t5_base_tok_ans_gen",
                                                                cache_dir="D:/cache",
                                                                device_map="cuda")
        # Variable assignment
        self.distractor_model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-race-Distractor")
        # Variable assignment
        self.distractor_tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-race-Distractor")

    # Function 'needs_answers' definition
    def needs_answers(self):
        # Conditional statement
        if self.model_type == 'race' or self.model_type == "squad":
            return False
        else:
            return True

    # Function 'get_contexts' definition
    def get_contexts(self, text, contexts_model_type="naive"):
        # Conditional statement
        if contexts_model_type == 'naive':
            # Variable assignment
            sentences = nltk.tokenize.sent_tokenize(text)

            # get steps of 3 sentences
            # Variable assignment
            prev = None
            # Loop starts here
            for i, sentence in enumerate(sentences):
                # Conditional statement
                if i + 1 != len(sentences):
                    # Variable assignment
                    after = sentences[i + 1]
                else:
                    # Variable assignment
                    after = None

                # Conditional statement
                if prev is not None and after is not None:
                    yield f"{prev} {sentence} {after}"

                # Variable assignment
                prev = sentence


        else:
            # todo: implement better, less naive context generating models
            raise NotImplementedError

    # Function 'compute_metrics' definition
    def compute_metrics(self, eval_pred):
        # Variable assignment
        logits, labels = eval_pred
        print(logits[1].shape)
        # Variable assignment
        predictions = np.argmax(logits[0], axis=-1)
        # Variable assignment
        predictions = [self.tokenizer.decode(prediction) for prediction in predictions]
        # Variable assignment
        labels = [self.tokenizer.decode(label) for label in labels]

        return self.metric.compute(predictions=predictions, references=labels)

    # Function 'tokenize_function' definition
    def tokenize_function(self, examples):
        # Variable assignment
        model_inputs = self.tokenizer(examples["sentence"], padding="max_length", truncation=True, return_tensors="pt")
        # Variable assignment
        labels = self.tokenizer(examples["labels"], padding="max_length", truncation=True, return_tensors="pt")
        # Variable assignment
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    # Function 'few_shot_train' definition
    def few_shot_train(self):
        # Conditional statement
        if self.model_type != "small":
            # Variable assignment
            data = load_dataset(path="./data_squad")
        else:
            # Variable assignment
            data = load_dataset(path="./data_base")
        # Variable assignment
        train_data = data["train"].map(self.tokenize_function, batched=True)
        # Variable assignment
        test_data = data["test"].map(self.tokenize_function, batched=True)
        # Conditional statement
        if self.model_type == "small":
            # Variable assignment
            epochs = 5
            # Variable assignment
            learning_rate = 0.00002
        else:
            # Variable assignment
            epochs = 10
            # Variable assignment
            learning_rate = 0.0001
        # Variable assignment
        training_arguments = Seq2SeqTrainingArguments(output_dir="test_trainer",
                                                      evaluation_strategy="epoch",
                                                      per_device_train_batch_size=1,
                                                      per_device_eval_batch_size=1,
                                                      num_train_epochs=epochs,
                                                      gradient_accumulation_steps=5,
                                                      gradient_checkpointing=True,
                                                      bf16=True,
                                                      learning_rate=learning_rate)
        # Variable assignment
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
    # Function 'get_distractors' definition
    def get_distractors(self, question, answer, context):
        # Variable assignment
        input_text = " ".join(
            [question, self.distractor_tokenizer.sep_token, answer, self.distractor_tokenizer.sep_token, context])
        # Variable assignment
        inputs = self.distractor_tokenizer(input_text, return_tensors="pt")
        # Variable assignment
        outputs = self.distractor_model.generate(**inputs, max_new_tokens=128)
        # Variable assignment
        distractors = self.distractor_tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Variable assignment
        distractors = distractors.replace(self.tokenizer.pad_token, "").replace(self.tokenizer.eos_token, "")
        # Variable assignment
        distractors = [y.strip() for y in distractors.split(self.tokenizer.sep_token)]
        return distractors

    # Function 'get_answer' definition
    def get_answer(self, context):
        # Variable assignment
        sents = context.split(". ")
        # Conditional statement
        if len(sents) == 3:
            # Variable assignment
            input_text = f"{sents[0]}. [HL] {sents[1]}. [HL] {sents[2]}."
            # Variable assignment
            input_text = input_text.strip()
            # Variable assignment
            inputs = self.answer_gen_tokenizer(input_text, max_length=512, pad_to_max_length=True,
                                               return_tensors='pt').to(
                "cuda")
            # Variable assignment
            output = self.answer_gen_model.generate(**inputs, max_length=100)
            # Variable assignment
            answer = self.answer_gen_tokenizer.decode(output[0], skip_special_tokens=True)
            print(answer.replace("[SEP]", "").strip())
            return answer.replace("[SEP]", "").strip()
        else:
            return None

    # Function 'generate' definition
    def generate(self, file, train=True):
        # Variable assignment
        text = ''

        # Checks the filetype and extracts text. Only works for .txt and .pdf files.
        # Conditional statement
        if file[-4:] == ".pdf":
            # Variable assignment
            pdfreader = PdfReader(file)
            # Loop starts here
            for page in pdfreader.pages:
                text += page.extract_text()
        # Conditional statement
        elif file[-4:] == ".txt":
            with open(file, 'r', encoding="utf-8") as f:
                # Variable assignment
                text = f.read()
        else:
            print("Invalid file type.")
            return

        # Prints the text after some special character removal.
        # Variable assignment
        text = re.sub("[^A-Za-z0-9 -_.,!?\"'/$# ()]", "", text)
        pprint(text)

        # few shot train
        # Conditional statement
        if train:
            self.few_shot_train()
            # Variable assignment
            self.model = self.model.to("cuda")

        # Variable assignment
        contexts = self.get_contexts(text)
        # Variable assignment
        output = []
        # Variable assignment
        i = 0
        # Loop starts here
        for context in contexts:  # remove indexing later, pair with testing of qnauestions and answers
            # Variable assignment
            answer = ""
            # Conditional statement
            if i >= 10:
                # Conditional statement
                break
            i += 1
            # Conditional statement
            if self.needs_answers():
                # Variable assignment
                answer = self.get_answer(context)
                # Conditional statement
                if answer is None:
                    continue
                print(answer)
                # Variable assignment
                context = f"answer: {answer}  context: {context} </s>"
            # Variable assignment
            inputs = self.tokenizer(context, return_tensors="pt", return_overflowing_tokens=False).to("cuda")
            # Variable assignment
            outputs = self.model.generate(**inputs, max_length=100)
            print(outputs[0])

            # Conditional statement
            if self.model_type != "small":
                # Variable assignment
                question_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                # Variable assignment
                question_answer = question_answer.replace(self.tokenizer.pad_token, "")\
                    .replace(self.tokenizer.eos_token, "")
                # Variable assignment
                question, answer = question_answer.split(self.tokenizer.sep_token)
                print(question_answer)
            else:
                # Variable assignment
                question_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                # Variable assignment
                question_answer = question_answer.replace(self.tokenizer.pad_token, "")\
                    .replace(self.tokenizer.eos_token, "")
                # Variable assignment
                question = question_answer.replace("question:", "").strip()

            # Variable assignment
            distractors = self.get_distractors(question, answer, context)
            output.append(f"Question: {question} Answer: {answer} Alt. Answers: {distractors}")

        # Variable assignment
        filename = file[:-4] + "_output.txt"
        # Conditional statement
        if train:
            # Variable assignment
            filename = filename[:-4] + "_trained.txt"
        # Conditional statement
        if self.model_type == "small":
            # Variable assignment
            filename = filename[:-4] + "_small.txt"

        with open(filename, 'w', encoding="utf-8") as f:
            f.write(str(output))

        torch.cuda.empty_cache()

        # Print output to console
        # pprint(output)


# Variable assignment
small_question_generator = QuestionGenerator("small")
small_question_generator.generate("Pilot_4.txt", train=False)
small_question_generator.generate("Pilot_4.txt")
del small_question_generator

# Variable assignment
squad_qgen = QuestionGenerator()
squad_qgen.generate("Pilot_2.txt", False)
squad_qgen.generate("Pilot_2.txt", True)
del squad_qgen

# Variable assignment
race_qgen = QuestionGenerator("race")
race_qgen.generate("Pilot_1.txt", False)
race_qgen.generate("Pilot_2.txt", True)