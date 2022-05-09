import numpy as np
from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss
from model.multimodal_bond.cross_attention import BertCrossAttnLayer, BertSelfAttnLayer, BertPooler
import os

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}

class RobertaForTokenClassification_v2(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # visual mapping
        self.vid2text_mapping = nn.Linear(config.video_embed_dim, config.hidden_size)
        # The cross-attention Layer
        self.visual_attention = BertCrossAttnLayer(config)
        # Self-attention Layers
        self.lang_self_att = BertSelfAttnLayer(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
        video_embedding=None,
        video_mask=None
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        final_embedding = outputs[0]
        sequence_output = self.dropout(final_embedding)

        # We borrow LXMERT's extended attention mask code
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Process the visual attention mask
        if video_mask is not None:
            extended_video_mask = video_mask.unsqueeze(1).unsqueeze(2)
            extended_video_mask = extended_video_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_video_mask = (1.0 - extended_video_mask) * -10000.0
        else:
            extended_video_mask = None

        # cross attention with video embeddings
        sequence_output = self.visual_attention(sequence_output, self.vid2text_mapping(video_embedding), extended_video_mask)
        sequence_output = self.lang_self_att(sequence_output, extended_attention_mask)
        # sequence_output = self.lang_self_att(sequence_output)

        logits = self.classifier(sequence_output)

        outputs = (logits, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:

            # Only keep active parts of the loss
            if attention_mask is not None or label_mask is not None:
                active_loss = True
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                if label_mask is not None:
                    active_loss = active_loss & label_mask.view(-1)
                active_logits = logits.view(-1, self.num_labels)[active_loss]


            if labels.shape == logits.shape:
                loss_fct = KLDivLoss()
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1, self.num_labels)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits, labels)
            else:
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


            outputs = (loss,) + outputs

        return outputs  # (loss), scores, final_embedding, (hidden_states), (attentions)


class RobertaForSequenceClassification_v2(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # The cross-attention Layer
        self.visual_attention = BertCrossAttnLayer(config)
        # Self-attention Layers
        self.lang_self_att = BertSelfAttnLayer(config)

        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        video_embedding=None,
        video_mask=None
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        final_embedding = outputs[0]
        sequence_output = self.dropout(final_embedding)

        # We borrow LXMERT's extended attention mask code
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Process the visual attention mask
        if video_mask is not None:
            extended_video_mask = video_mask.unsqueeze(1).unsqueeze(2)
            extended_video_mask = extended_video_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_video_mask = (1.0 - extended_video_mask) * -10000.0
        else:
            extended_video_mask = None

        # cross attention with video embeddings
        sequence_output = self.visual_attention(sequence_output, video_embedding, extended_video_mask)
        sequence_output = self.lang_self_att(sequence_output, extended_attention_mask)
        # sequence_output = self.lang_self_att(sequence_output)

        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, final_embedding, (hidden_states), (attentions)


class RobertaForTokenClassificationLateFusion(BertPreTrainedModel):

    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # base of the model
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self attention + text-based outputs [auxilliary loss is optional]
        self.text_self_attention_1 = BertSelfAttnLayer(config)
        self.aux_classifier = nn.Linear(config.hidden_size, config.num_labels)

        # visual mapping
        self.vid2text_mapping = nn.Linear(config.video_embed_dim, config.hidden_size)
        self.vid_self_attention_1 = BertSelfAttnLayer(config)

        # multimodal interactions
        # self attention for transforming text
        self.text_self_attention_2 = BertSelfAttnLayer(config)
        # The cross-attention Layers
        self.vid2text_attention = BertCrossAttnLayer(config)
        self.text2vid_attention = BertCrossAttnLayer(config)
        self.text2text_attention = BertCrossAttnLayer(config)
        # final self attention layer for text
        self.text_self_attention_3 = BertCrossAttnLayer(config)
        # visual gate
        self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size)

        # self.fused_remapping = nn.Linear(config.hidden_size*2, config.hidden_size)
        # self.fused_self_attention = BertSelfAttnLayer(config)

        # Self-attention Layers
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
        video_embedding=None,
        video_mask=None
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        final_embedding = outputs[0]
        sequence_output = self.dropout(final_embedding)

        # We borrow LXMERT's extended attention mask code
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Process the visual attention mask
        if video_mask is not None:
            extended_video_mask = video_mask.unsqueeze(1).unsqueeze(2)
            extended_video_mask = extended_video_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_video_mask = (1.0 - extended_video_mask) * -10000.0
        else:
            extended_video_mask = None

        # After BERT/RoBERTa, the first step is self attention 1
        text_output = self.text_self_attention_1(sequence_output, extended_attention_mask)

        # Multimodal Interactions
        video_embedding = self.vid_self_attention_1(self.vid2text_mapping(video_embedding))
        sequence_output = self.text_self_attention_2(sequence_output, extended_attention_mask)

        # cross attention with video embeddings
        cross_output_1, _ = self.text2vid_attention(sequence_output, video_embedding, extended_video_mask, return_attention=True)
        # if os.path.exists('text2vid_attention.npy'):
        #     temp = np.load('text2vid_attention.npy')
        #     temp = np.concatenate((temp, text2vid_attn_scores.cpu().data.numpy()), axis=0)
        #     np.save('text2vid_attention.npy', temp)
        # else:
        #     np.save('text2vid_attention.npy', text2vid_attn_scores.cpu().data.numpy())

        # cross attention with text embeddings
        cross_output_2, _ = self.vid2text_attention(video_embedding, sequence_output, extended_attention_mask, return_attention=True)
        # if os.path.exists('vid2text_attention.npy'):
        #     temp = np.load('vid2text_attention.npy')
        #     temp = np.concatenate((temp, vid2text_attn_scores.cpu().data.numpy()), axis=0)
        #     np.save('vid2text_attention.npy', temp)
        # else:
        #     np.save('vid2text_attention.npy', vid2text_attn_scores.cpu().data.numpy())

        cross_output_3 = self.text2text_attention(sequence_output, cross_output_2, extended_video_mask)

        # sequence_output = self.lang_self_att(sequence_output)
        merged_representation = torch.cat((cross_output_1, cross_output_3), dim=-1)
        gate_values = torch.sigmoid(self.gate(merged_representation))  # batch_size, text_len, hidden_dim
        # if os.path.exists('gate_values.npz'):
        #     temp1 = np.load('gate_values.npz')
        #     temp2 = gate_values.cpu().data.numpy()
        #     means = np.concatenate([temp1['mean'], np.mean(temp2, axis=-1)], axis=0)
        #     stds = np.concatenate([temp1['std'], np.std(temp2, axis=-1)], axis=0)
        #     np.savez('gate_values.npz', mean=means, std=stds)
        # else:
        #     temp2 = gate_values.cpu().data.numpy()
        #     np.savez('gate_values.npz', mean=np.mean(temp2, axis=-1), std=np.std(temp2, axis=-1))

        gated_multimodal_output = torch.mul(gate_values, cross_output_1)

        # fused_output = self.fused_remapping(torch.cat((text_output, gated_multimodal_output), dim=-1))
        # fused_output = self.fused_self_attention(fused_output)

        # logits = self.classifier(fused_output)
        logits = self.classifier(torch.cat((text_output, gated_multimodal_output), dim=-1))

        outputs = (logits, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:

            # Only keep active parts of the loss
            if attention_mask is not None or label_mask is not None:
                active_loss = True
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                if label_mask is not None:
                    active_loss = active_loss & label_mask.view(-1)
                active_logits = logits.view(-1, self.num_labels)[active_loss]


            if labels.shape == logits.shape:
                loss_fct = KLDivLoss()
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1, self.num_labels)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits, labels)
            else:
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


            outputs = (loss,) + outputs

        return outputs  # (loss), scores, final_embedding, (hidden_states), (attentions)
