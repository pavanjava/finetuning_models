import sys
from os import truncate

data = [
    {"prompt": "What is cardiology?",
     "chosen": "Cardiology is a branch of medicine that deals with disorders of the heart and the cardiovascular system.",
     "rejected": "Cardiology is the study of the lungs and respiratory system."},
    {"prompt": "Define myocardial infarction.",
     "chosen": "A myocardial infarction, commonly known as a heart attack, occurs when blood flow decreases or stops in one of the arteries of the heart, causing tissue death to the heart muscle.",
     "rejected": "Myocardial infarction is a condition where the heart beats too fast."},
    {"prompt": "What is atrial fibrillation?",
     "chosen": "Atrial fibrillation is the most commonly encountered arrhythmia in clinical practice, characterized by irregular and often rapid heartbeat.",
     "rejected": "Atrial fibrillation is a blockage in the coronary arteries."},
    {"prompt": "Explain heart failure.",
     "chosen": "Heart failure is a condition in which the heart cannot pump enough blood to meet the needs of the body's other organs, though the heart keeps pumping, just not as effectively.",
     "rejected": "Heart failure means the heart has completely stopped beating."},
    {"prompt": "What is cardiac catheterization?",
     "chosen": "Cardiac catheterization is a procedure where a thin flexible tube called a catheter is inserted into an artery or vein and guided to the heart to diagnose problems in heart function, blockages, and valve conditions.",
     "rejected": "Cardiac catheterization is a surgery to replace heart valves."},
    {"prompt": "Define coronary artery disease.",
     "chosen": "Coronary artery disease is a condition caused by the buildup of plaque in the coronary arteries, leading to reduced blood flow to the heart muscle.",
     "rejected": "Coronary artery disease is an infection of the heart chambers."},
    {"prompt": "What is an echocardiogram?",
     "chosen": "An echocardiogram is a noninvasive diagnostic test that uses sound waves (ultrasound) to create real-time images of the heart's structure and function, including chambers, valves, walls, and blood flow patterns.",
     "rejected": "An echocardiogram is a blood test that measures cholesterol levels."},
    {"prompt": "Explain cardiomyopathy.",
     "chosen": "Cardiomyopathy is a disease of the heart muscle that leads to generalized deterioration of the muscle and its pumping ability.",
     "rejected": "Cardiomyopathy is a procedure to open blocked arteries."},
    {"prompt": "What causes ventricular arrhythmias after myocardial infarction?",
     "chosen": "Ventricular arrhythmias after myocardial infarction are caused by tissue death and myocardial scarring that alter the normal conduction pathways of the heart, creating a heterogeneous environment that promotes reentry and abnormal electrical activity.",
     "rejected": "Ventricular arrhythmias are caused by drinking too much water."},
    {"prompt": "What is the relationship between heart failure and myocardial infarction?",
     "chosen": "Heart failure is a frequent complication of myocardial infarction, developing in approximately 15-20% of patients during hospital admission and 10-12% after discharge, caused by factors such as infarct size, ventricular remodeling, and cardiomyocyte loss.",
     "rejected": "Heart failure and myocardial infarction are completely unrelated conditions."},
    {"prompt": "Define ejection fraction.",
     "chosen": "Ejection fraction is a measurement of how well the heart muscle contracts and pumps blood, with a normal range of 50% to 70%.",
     "rejected": "Ejection fraction is the number of times the heart beats per minute."},
    {"prompt": "What is a cardiac pacemaker?",
     "chosen": "A cardiac pacemaker is a surgically implanted device that maintains a steady heartbeat in people with a slow or irregular heartbeat.",
     "rejected": "A cardiac pacemaker is a medication used to lower blood pressure."},
    {"prompt": "Explain coronary angiography.",
     "chosen": "Coronary angiography is a diagnostic procedure where a catheter is used to inject dye into the coronary arteries and X-ray pictures are taken to reveal areas that are clogged or narrowed.",
     "rejected": "Coronary angiography is a type of heart surgery to repair valves."},
    {"prompt": "What is ischemic cardiomyopathy?",
     "chosen": "Ischemic cardiomyopathy refers to systolic left ventricular dysfunction in the setting of obstructive coronary artery disease and represents the most common cause of heart failure worldwide.",
     "rejected": "Ischemic cardiomyopathy is a viral infection of the heart."},
    {"prompt": "Define ventricular fibrillation.",
     "chosen": "Ventricular fibrillation is an abnormal irregular heart rhythm characterized by very rapid uncoordinated fluttering contractions of the lower chambers (ventricles) of the heart, disrupting the synchrony between heartbeat and pulse.",
     "rejected": "Ventricular fibrillation is when the heart beats too slowly."},
    {"prompt": "What is the purpose of a stress test?",
     "chosen": "A stress test shows the changes in heart rhythm when resting and exercising, helping diagnose coronary artery disease and assess heart function under physical stress.",
     "rejected": "A stress test measures the amount of stress hormones in the blood."},
    {"prompt": "Explain cardiac ablation.",
     "chosen": "Cardiac ablation is a procedure used to correct abnormal heart rhythms such as atrial fibrillation by using radiofrequency energy to destroy small areas of heart tissue causing the arrhythmia.",
     "rejected": "Cardiac ablation is a medication to thin the blood."},
    {"prompt": "What is an implantable cardioverter defibrillator (ICD)?",
     "chosen": "An implantable cardioverter defibrillator (ICD) is a device that monitors the heart for fast rhythms and delivers an electrical shock to reset normal rhythm when dangerous arrhythmias are detected.",
     "rejected": "An ICD is a type of artificial heart valve."},
    {"prompt": "Define angina pectoris.",
     "chosen": "Angina pectoris is retrosternal chest pain or discomfort that classically radiates to the left shoulder, arm, or jaw, typically caused by reduced blood flow to the heart muscle.",
     "rejected": "Angina pectoris is inflammation of the lung lining."},
    {"prompt": "What is ventricular remodeling after myocardial infarction?",
     "chosen": "Ventricular remodeling after myocardial infarction is a process involving changes in ventricular geometry, wall thinning, and progressive cardiomyocyte loss triggered by cardiomyocyte death, scar formation, and chronic neurohumoral activation.",
     "rejected": "Ventricular remodeling is the natural healing process that always improves heart function."},
    {"prompt": "Explain transesophageal echocardiography.",
     "chosen": "Transesophageal echocardiography (TEE) uses an ultrasound probe placed in the patient's esophagus to provide a more direct view of heart structures, particularly useful for detecting thrombi and guiding cardioversion.",
     "rejected": "Transesophageal echocardiography is an X-ray of the chest."},
    {"prompt": "What is the role of B-type natriuretic peptide (BNP) in cardiology?",
     "chosen": "B-type natriuretic peptide (BNP) is a cardiac biomarker that is evaluated to investigate for underlying heart disease, particularly heart failure.",
     "rejected": "BNP is a vitamin supplement for heart health."},
    {"prompt": "Define percutaneous coronary intervention.",
     "chosen": "Percutaneous coronary intervention (PCI) is a minimally invasive procedure performed by cardiologists to open blocked coronary arteries, typically involving angioplasty and stenting.",
     "rejected": "Percutaneous coronary intervention is open-heart surgery."},
    {"prompt": "What is cardiogenic shock?",
     "chosen": "Cardiogenic shock is a condition where the heart is unable to adequately pump blood, most commonly occurring after an acute myocardial infarction and representing the largest cause of in-hospital mortality.",
     "rejected": "Cardiogenic shock is an allergic reaction to heart medications."},
    {"prompt": "Explain left ventricular hypertrophy.",
     "chosen": "Left ventricular hypertrophy (LVH) is a thickening of the heart muscle in the left ventricle, often associated with conditions like hypertension and can be detected on electrocardiogram.",
     "rejected": "Left ventricular hypertrophy is the enlargement of the right side of the heart."},
    {"prompt": "What is a Holter monitor?",
     "chosen": "A Holter monitor is a portable device that continuously records a patient's heart rhythm and electrical activity over an extended period, typically 24 to 48 hours, to detect arrhythmias that may not be captured during brief ECGs.",
     "rejected": "A Holter monitor is a device that measures blood pressure throughout the day."},
    {"prompt": "Define coronary artery bypass grafting (CABG).",
     "chosen": "Coronary artery bypass grafting (CABG) is a surgical procedure performed by cardiac surgeons to create new routes for blood flow around blocked coronary arteries using grafts from other blood vessels.",
     "rejected": "CABG is a non-invasive procedure to dissolve blood clots."},
    {"prompt": "What are the common symptoms of myocardial infarction?",
     "chosen": "Common symptoms of myocardial infarction include retrosternal chest pain radiating to the left shoulder, arm, or jaw, shortness of breath, nausea, feeling faint, cold sweat, and fatigue, though about 30% of patients have atypical symptoms.",
     "rejected": "The only symptom of myocardial infarction is severe headache."},
    {"prompt": "Explain fractional flow reserve (FFR).",
     "chosen": "Fractional flow reserve (FFR) is a measurement performed during cardiac catheterization using adenosine to induce maximum hyperemia, helping guide clinical decision-making for patients with moderate coronary artery stenosis.",
     "rejected": "FFR is a blood test to check kidney function."},
    {"prompt": "What is the significance of troponin levels in cardiac diagnosis?",
     "chosen": "Troponins are cardiac biomarkers that become elevated when there is damage to heart muscle, such as during myocardial infarction, and are essential for diagnosing acute coronary syndromes.",
     "rejected": "Troponin levels measure blood sugar in diabetic patients."}
]

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

# Load the base model
base_model = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(base_model)
model.resize_token_embeddings(len(tokenizer))
model.train()

# Format the data as per SFT
# Format data for SFT
sft_data = [
    {
        "input_ids": tokenizer(d["prompt"] + d["chosen"], truncation=True, padding="max_length", max_length=128,
                               return_tensors="pt")["input_ids"].squeeze(),
        "labels": tokenizer(d["prompt"] + d["chosen"], truncation=True, padding="max_length", max_length=128,
                            return_tensors="pt")["input_ids"].squeeze()
    }
    for d in data
]

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

sft_dataset = SFTDataset(sft_data)

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=10,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sft_dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./sft_finetuned_model")
tokenizer.save_pretrained("./sft_finetuned_model")
