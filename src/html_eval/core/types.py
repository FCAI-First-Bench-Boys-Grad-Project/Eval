from typing import Union 

class Sample(dict):
    """
    A single sample from the dataset.
    Acts like both a dict and an object with attributes.
    """
    id: str
    content: str 
    query: str
    ground_truth: str
    is_content_url: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

        if "is_content_url" not in kwargs:
            content = kwargs.get("content", "")
            self.is_content_url = isinstance(content, str) and bool(str(content).lower().startswith(('http://', 'https://')))
        else:
            self.is_content_url = kwargs["is_content_url"]

    def __getitem__(self, key):
        # tuple-style indexing (by position)
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__[key] = value  # keep attributes in sync

    def __setattr__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)



class SamplePrediction(dict):
    """
    A single prediction corresponding to a Sample.
    Acts like both a dict and an object with attributes.
    """
    id: str
    query: str
    ground_truth: str
    prediction: Union[str, dict]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        # tuple-style indexing (by position)
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__[key] = value  # keep attributes in sync

    def __setattr__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

class SampleEvaluation(SamplePrediction):
    """
    A single evaluation result for a SamplePrediction.
    Extends SamplePrediction with additional evaluation metrics.
    """
    evaluation: dict
