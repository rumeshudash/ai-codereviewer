import { readFileSync } from "fs";
import * as core from "@actions/core";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAI } from "@ai-sdk/openai";
import { generateText, Output } from "ai";
import { Octokit } from "@octokit/rest";
import minimatch from "minimatch";
import parseDiff, { File } from "parse-diff";
import { z } from "zod";

const GITHUB_TOKEN: string = core.getInput("GITHUB_TOKEN");
const OPENAI_API_KEY: string = core.getInput("OPENAI_API_KEY");
const OPENAI_API_MODEL: string = core.getInput("OPENAI_API_MODEL");
const GEMINI_API_KEY: string = core.getInput("GEMINI_API_KEY");
const GEMINI_MODEL: string = core.getInput("GEMINI_MODEL");

const openaiProvider = OPENAI_API_KEY
  ? createOpenAI({ apiKey: OPENAI_API_KEY })
  : null;
const googleProvider = GEMINI_API_KEY
  ? createGoogleGenerativeAI({ apiKey: GEMINI_API_KEY })
  : null;

const model = openaiProvider
  ? openaiProvider(OPENAI_API_MODEL)
  : googleProvider
  ? googleProvider(GEMINI_MODEL)
  : (() => {
      core.setFailed(
        "Either OPENAI_API_KEY or GEMINI_API_KEY must be set. Provide one to run the code review."
      );
      process.exit(1);
    })();

const octokit = new Octokit({ auth: GITHUB_TOKEN });

const reviewOutputSchema = z.object({
  reviews: z
    .array(
      z.object({
        path: z.string(),
        lineNumber: z.union([z.string(), z.number()]),
        endLineNumber: z.union([z.string(), z.number()]).optional(),
        reviewComment: z.string(),
      })
    )
    .optional(),
});

interface PRDetails {
  owner: string;
  repo: string;
  pull_number: number;
  title: string;
  description: string;
}

async function getPRDetails(): Promise<PRDetails> {
  const { repository, number } = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH || "", "utf8")
  );
  const prResponse = await octokit.pulls.get({
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
  });
  return {
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
    title: prResponse.data.title ?? "",
    description: prResponse.data.body ?? "",
  };
}

async function getDiff(
  owner: string,
  repo: string,
  pull_number: number
): Promise<string | null> {
  const response = await octokit.pulls.get({
    owner,
    repo,
    pull_number,
    mediaType: { format: "diff" },
  });
  // @ts-expect-error - response.data is a string
  return response.data;
}

type ReviewCommentInput = {
  body: string;
  path: string;
  line: number;
  side: "LEFT" | "RIGHT";
  start_line?: number;
  start_side?: "LEFT" | "RIGHT";
};

async function analyzeCode(
  parsedDiff: File[],
  prDetails: PRDetails
): Promise<ReviewCommentInput[]> {
  const prompt = createPromptForAllDiffs(parsedDiff, prDetails);
  console.log("Prompt:", prompt);
  const aiResponse = await getAIResponse(prompt);
  console.log("AI Response:", aiResponse);
  if (!aiResponse || aiResponse.length === 0) return [];
  const comments = createCommentsFromResponse(aiResponse);
  return comments.filter((c) => c.body.length > 0);
}

function createPromptForAllDiffs(
  parsedDiff: File[],
  prDetails: PRDetails
): string {
  const diffSections = parsedDiff
    .filter((file) => file.to !== "/dev/null")
    .map((file) => {
      const chunksText = file.chunks
        .map(
          (chunk) =>
            `${chunk.content}\n${chunk.changes
              // @ts-expect-error - ln and ln2 exist where needed
              .map((c) => `${c.ln ? c.ln : c.ln2} ${c.content}`)
              .join("\n")}`
        )
        .join("\n");
      return `## File: ${file.to}\n\`\`\`diff\n${chunksText}\n\`\`\``;
    })
    .join("\n\n");

  return `You are a strict code reviewer. Review the pull request diff and output ONLY actionable issues. Instructions:

CRITICAL - When to output comments:
- If the code has no bugs, security issues, style problems, or clear improvements to suggest, you MUST return exactly: {"reviews": []}
- Only provide a review if there is a concrete, fixable problem or a clear improvement. If in doubt, return an empty review array.
- Only add a review when there is a concrete, fixable problem or a clear improvement (e.g. bug, security, wrong logic, missing error handling, misleading name). If in doubt, return empty reviews.

Output format (use this exact JSON shape):
- When there are issues: {"reviews": [{"path": "<exact file path from diff>", "lineNumber": <line>, "endLineNumber": <optional>, "reviewComment": "<markdown comment>"}]}
- When there are no issues: {"reviews": []}

Rules:
- "path" MUST match the file path from the diff exactly (e.g. "src/main.ts").
- Use lineNumber for the line to comment on (or last line of a range). Use endLineNumber only for a multi-line range; omit for single-line.
- Avoid suggestions for refactoring unless they address a significant performance, security, or maintainability issue.
- Write comments in GitHub Markdown format.

Pull request title: ${prDetails.title}
Pull request description:

---
${prDetails.description}
---

Full diff to review (multiple files):

${diffSections}
`;
}

type AIReviewItem = {
  path: string;
  lineNumber: string;
  endLineNumber?: string;
  reviewComment: string;
};

async function getAIResponse(prompt: string): Promise<AIReviewItem[] | null> {
  try {
    const { output } = await generateText({
      model,
      system: prompt,
      prompt: "",
      maxOutputTokens: 4000,
      temperature: 0.2,
      output: Output.object({
        schema: reviewOutputSchema,
        name: "CodeReview",
        description:
          "List of code review comments. Must be an empty array when there are no issues to report.",
      }),
    });

    const raw = output.reviews ?? [];
    const reviews = raw.map((r) => ({
      path: r.path,
      lineNumber: String(r.lineNumber),
      endLineNumber:
        r.endLineNumber != null ? String(r.endLineNumber) : undefined,
      reviewComment: (r.reviewComment || "").trim(),
    }));
    return reviews;
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

function createCommentsFromResponse(
  aiResponses: AIReviewItem[]
): ReviewCommentInput[] {
  return aiResponses.flatMap((aiResponse) => {
    const line = Number(aiResponse.lineNumber);
    const endLine =
      aiResponse.endLineNumber != null
        ? Number(aiResponse.endLineNumber)
        : undefined;
    const isMultiLine = endLine != null && endLine !== line;
    const [startLine, lastLine] =
      isMultiLine && endLine != null
        ? endLine < line
          ? [endLine, line]
          : [line, endLine]
        : [line, line];

    const comment: ReviewCommentInput = {
      body: aiResponse.reviewComment,
      path: aiResponse.path,
      line: lastLine,
      side: "RIGHT",
    };
    if (isMultiLine) {
      comment.start_line = startLine;
      comment.start_side = "RIGHT";
    }
    return comment;
  });
}

async function createReviewComment(
  owner: string,
  repo: string,
  pull_number: number,
  comments: ReviewCommentInput[]
): Promise<void> {
  await octokit.pulls.createReview({
    owner,
    repo,
    pull_number,
    comments,
    event: "COMMENT",
  });
}

async function main() {
  const prDetails = await getPRDetails();
  let diff: string | null;
  const eventData = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH ?? "", "utf8")
  );

  if (eventData.action === "opened") {
    diff = await getDiff(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number
    );
  } else if (eventData.action === "synchronize") {
    const newBaseSha = eventData.before;
    const newHeadSha = eventData.after;

    const response = await octokit.repos.compareCommits({
      headers: {
        accept: "application/vnd.github.v3.diff",
      },
      owner: prDetails.owner,
      repo: prDetails.repo,
      base: newBaseSha,
      head: newHeadSha,
    });

    diff = String(response.data);
  } else {
    console.log("Unsupported event:", process.env.GITHUB_EVENT_NAME);
    return;
  }

  if (!diff) {
    console.log("No diff found");
    return;
  }

  const parsedDiff = parseDiff(diff);

  const excludePatterns = core
    .getInput("exclude")
    .split(",")
    .map((s) => s.trim());

  const filteredDiff = parsedDiff.filter((file) => {
    return !excludePatterns.some((pattern) =>
      minimatch(file.to ?? "", pattern)
    );
  });

  const comments = await analyzeCode(filteredDiff, prDetails);
  if (comments.length > 0) {
    await createReviewComment(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
      comments
    );
  }
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
