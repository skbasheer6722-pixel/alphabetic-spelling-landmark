# Project details for report

**Title:** Design and Development of a Working Prototype for ASL Alphabet Fingerspelling Recognition Using a CNN Classifier on Hand-Landmark Images

**Abstract:** American Sign Language (ASL) is an important means of communication for deaf and hard-of-hearing individuals. However, communication between sign language users and people unfamiliar with ASL can sometimes be challenging. This project develops a prototype for recognizing static ASL alphabet fingerspelling handshapes using a Convolutional Neural Network (CNN) and hand-landmark images. A webcam captures hand gestures, while MediaPipe detects 21 hand landmarks. The coordinates are normalized and rendered as a compact image representation, which is classified by a CNN trained on labeled samples collected for each supported letter. The predicted letter and confidence score are displayed alongside the live camera feed. The system includes data collection, preprocessing, model training, hand-landmark detection, gesture classification and real-time display. Performance is assessed using accuracy, precision, recall, F1-score and a confusion matrix. The prototype demonstrates the use of computer vision and deep learning for accessible human-computer interaction. The current implementation handles static handshapes; motion-based letters such as J and Z require a temporal extension.

**Keywords:** American Sign Language (ASL); fingerspelling; recognition; convolutional neural network (CNN); MediaPipe; hand-landmark detection; computer vision; deep learning; image preprocessing; real-time gesture recognition; human-computer interaction.

**Suggested modules:** Data acquisition, landmark extraction, normalization/rendering, CNN training, real-time inference, evaluation.

**Suggested future enhancement:** Sequence capture and an LSTM/GRU/3D-CNN branch for motion-dependent letters J and Z; signer-independent testing; sentence-level fingerspelling support.
