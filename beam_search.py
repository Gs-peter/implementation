import torch.nn.functional as F
import torch


class beamsearch:
    def __init__(
            self,
            top_k,
            tokenizer,
            max_length_limit
            ):
        self.top_k = top_k
        self.tokenizer = tokenizer
        self.max_length_limit = max_lenth_limit
        self.global_view = []

    def beamsearch(
                self,
                input_ids_ ,
                current_gen ,
                eos_token = self.tokenizer.eos_token,
                tracking_token = (),
                tracking_prob = ()
            ):
        if current_gen in eos_token:
            self.global_view.append(
                    (
                        tracking_token,
                        tracking_prob
                        )
                    )
            break

        next_token_logsigmoid = F.log_sigmoid(
                model(
                    **input_ids_
                    ).logits
                )
        top_k_mask = next_token_logsigmoid > top_k
        top_k_prob = next_token_logsigmoid[ top_k_mask ]


        for token_id, more_like_token in enumerate(
                top_k_prob
                ):
            if more_like_token != 0:
                tracking_token = tracking_token.add(token_id)
                tracking_prob = tracking_prob.add(token_prob)

                self.beamsearch(
                        input_ids_ = input_ids_.add(
                            token_id
                            ),
                        current_gen = token_id,
                        tracking_token = tracking_token,
                        tracking_prob = tracking_prob
                        )
    def __call__(self):

        highest_response = sorted(
                self.global_view,
                key = lambda x : x[1],
                reverse = True
                )
        return highest_response[0]




