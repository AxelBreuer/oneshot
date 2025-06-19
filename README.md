# oneshot
A one-shot inpainting algorithm based on the topological asymptotic analysis

# Background
A (very) long time ago, as a student at INSA Toulouse, I took a course on shape optimization taught by Philippe Guillaume and Mohamed Masmoudi. I was fascinated by the elegance of the topic, particularly the “topological gradient.” The analytical calculations were involved, and the theoretical justifications even more so. Yet, according to the slides, you could eventually compute an optimal shape in a super-fast way. Unfortunately, I left the course feeling a bit frustrated, since I never had the opportunity to implement that gradient or apply it to a toy problem.

Many years later, I came across the article *A One-Shot Inpainting Algorithm Based on Topological Asymptotic Analysis* by Didier Auroux and Mohamed Masmoudi. Finally, here was an application of the topological gradient that anyone could try at home. I decided to give it a go and implement the algorithm described in their paper, which led me to contact Didier Auroux—he was kind enough to share his insights on the topic.

The code in this repo attempts to reproduce their published results.

# Installation

**Step 1** Clone the oneshot repository into your sources directory, in our example `C:\Users\<your name>\src`.


---

*Troubleshooting: If you don't have git on your machine to pull the repository, download and install https://gitforwindows.org/. Then, in your freshly installed Git Bash type*

```console
cd /C/Users/<your GAIA>/src
git clone https://github.com/AxelBreuer/oneshot.git
```

---


**Step 2** At this stage, you should now have a subdirectory `C:\Users\<your name>\src\oneshot`.

In your Anaconda prompt, type

```console
cd C:\Users\<your name>\src\oneshot
python -m pip install -e .
```
