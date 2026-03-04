


class RecursiveWeightedSum:
    def __init__(self, coeffs, init_values, length, reverse_output_order=False):
        self.coeffs = coeffs
        self.init_values = init_values
        self.num_past_values = len(self.coeffs)
        self.length = length
        self.variable_name = "x"
        self.length_name = "n"
        self.sep = ", "
        self.eq = " = "
        self.reverse_output_order = reverse_output_order

    def compute(self):
        assert len(self.init_values) == len(self.coeffs), "Length of init_values must match length of coeffs"

        out = [*self.init_values]

        for i in range(self.length - self.num_past_values):
            out += [sum(c * v for c, v in zip(self.coeffs, reversed(out[-self.num_past_values:])))]

        if self.reverse_output_order:
            out = out[::-1]

        return out

    def get_term(self, index, coeff=None):
        text = ""

        if coeff is not None and coeff != 1:
            text += f"{coeff}*"

        if index is None:
            text += f"{self.variable_name}_{self.length_name}"
        elif index >= 0:
            text += f"{self.variable_name}_{index}"
        else:
            text += f"{self.variable_name}_{{{self.length_name}-{-index}}}"

        return text

    def check_correct(self, x):
        return isinstance(x, list) and x == self.compute()

    def get_def_desc(self):
        text = ""

        for i, init_value in enumerate(self.init_values):
            text += self.get_term(i) + self.eq + str(init_value) + self.sep

        text += self.get_term(None) + self.eq + " + ".join(self.get_term(-(i + 1), coeff=self.coeffs[i]) for i in range(len(self.init_values)))
        text = text.replace("+ -", "- ")

        return text

    def get_output_desc(self):
        text = "["

        for i in (range(self.length) if not self.reverse_output_order else range(self.length - 1, -1, -1)):
            text += self.get_term(i)

            if i != (self.length - 1 if not self.reverse_output_order else 0):
                text += self.sep

        text += "]"

        return text
