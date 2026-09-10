const API_BASE = '/api/v1';

import type { WorkflowDocument } from './AgentEditor';

export type BrainType = 'tiny_llm' | 'test' | 'clarification';

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface LLMStatusResponse {
  base_url: string;
  model: string;
  available: boolean;
  models: string[];
  message: string;
  managed?: boolean;
  context_size?: number | null;
}

export interface LocalModelProfile {
  id: string;
  label: string;
  path: string;
  size_bytes: number;
  is_instruction_tuned: boolean;
  active: boolean;
}

export interface LocalModelsResponse {
  directory: string;
  models: LocalModelProfile[];
  devices: LocalExecutionDevice[];
  active: { base_url: string; model: string; managed: boolean; device?: string; context_size?: number | null };
}

export interface LocalExecutionDevice {
  id: 'cpu' | 'gpu';
  label: string;
  available: boolean;
  detail: string;
}

export interface TeachingWorkspaceTreeEntry {
  name: string;
  kind: 'directory' | 'file';
  relative_path: string;
  size_bytes?: number;
  readable?: boolean;
  children?: TeachingWorkspaceTreeEntry[];
}

export interface TeachingWorkspaceTree {
  root_path: string;
  name: string;
  entries: TeachingWorkspaceTreeEntry[];
  entry_count: number;
  truncated: boolean;
  max_depth: number;
  max_entries: number;
}

export interface TeachingWorkspaceSelection {
  selected: boolean;
  root_path: string;
  tree: TeachingWorkspaceTree | null;
}

export interface ModelToolCallMetric {
  provider: string;
  model: string;
  runs: number;
  completed_runs: number;
  decisions: number;
  decision_errors: number;
  decision_error_breakdown: Record<string, number>;
  tool_calls_requested: number;
  tool_calls_validated: number;
  tool_calls_started: number;
  tool_calls_succeeded: number;
  tool_calls_failed: number;
  tool_calls_blocked: number;
  tool_execution_success_rate: number | null;
  proposal_validation_rate: number | null;
  plan_to_tool_success_rate: number | null;
  last_event_at: number;
}

export interface ModelToolMetricsResponse {
  reset_at: number;
  models: ModelToolCallMetric[];
}


export interface AgentResponse {
  agent_id: string;
  name: string;
  description?: string | null;
  brain: {
    type: BrainType;
    model?: string | null;
    temperature: number;
    max_attempts: number;
  };
  tools: unknown[];
}

export interface CaseTemplate {
  case_id: string;
  title: string;
  description: string;
  objective: string;
  input: string;
  brain_type: BrainType;
  brain_model?: string | null;
  tools: string[];
}

export interface AgentEvent {
  event_id: string;
  run_id: string;
  step: number;
  type: string;
  timestamp: number;
  node: string;
  data: Record<string, unknown>;
}

export interface RunResponse {
  run_id: string;
  agent_id: string;
  input: string;
  status: 'created' | 'running' | 'paused' | 'waiting_user' | 'completed' | 'failed';
  step: number;
  failures: number;
  decision_failures: number;
  tool_failures: number;
  policy_blocks: number;
  memory: Record<string, unknown>;
  context: Array<Record<string, unknown>>;
  output?: string | null;
  error?: string | null;
  waiting_question?: string | null;
  waiting_choices?: WaitingChoice[];
  waiting_response_schema?: WaitingResponseSchema | null;
  events: AgentEvent[];
}

export interface WaitingChoice {
  file_id: string;
  name: string;
  directory?: string;
}

export interface WaitingResponseSchema {
  type: 'file_selection';
  source_observation: string;
  candidate_file_ids: string[];
  allow_custom_input: boolean;
  selection_mode?: 'single' | 'multiple';
  min_selections?: number;
  max_selections?: number;
}

export interface ReplayResponse {
  run: Omit<RunResponse, 'events'>;
  events: AgentEvent[];
  event_count: number;
  step_count: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(API_BASE + path, init);
  if (!response.ok) {
    let detail = 'request failed: ' + response.status;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      // Keep the HTTP status when the server did not return JSON.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health');
}

export async function getLLMStatus(): Promise<LLMStatusResponse> {
  return request<LLMStatusResponse>('/llm/status');
}

export async function getLocalModels(): Promise<LocalModelsResponse> {
  return request<LocalModelsResponse>('/local-models');
}

export async function selectTeachingWorkspaceDirectory(): Promise<TeachingWorkspaceSelection> {
  return request<TeachingWorkspaceSelection>('/teaching-workspace/select-directory', { method: 'POST' });
}

export async function getTeachingWorkspaceTree(rootPath: string): Promise<TeachingWorkspaceTree> {
  return request<TeachingWorkspaceTree>('/teaching-workspace/tree?root_path=' + encodeURIComponent(rootPath));
}

export async function activateLocalModel(modelId: string, device = 'cpu', contextSize?: number): Promise<LocalModelsResponse> {
  await request<{ profile: LocalModelProfile }>('/local-models/' + encodeURIComponent(modelId) + '/activate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ device, context_size: contextSize }),
  });
  return getLocalModels();
}

export async function getModelToolMetrics(): Promise<ModelToolMetricsResponse> {
  return request<ModelToolMetricsResponse>('/metrics/model-tool-calls');
}

export async function clearModelToolMetrics(): Promise<ModelToolMetricsResponse> {
  return request<ModelToolMetricsResponse>('/metrics/model-tool-calls', { method: 'DELETE' });
}

export async function getCases(): Promise<CaseTemplate[]> {
  return request<CaseTemplate[]>('/cases');
}

export interface CreateCaseTemplate {
  case_id: string;
  title: string;
  description: string;
  objective: string;
  input: string;
  brain_type: BrainType;
  brain_model?: string | null;
  tools: string[];
}

export async function createCaseTemplate(
  template: CreateCaseTemplate,
): Promise<CaseTemplate> {
  return request<CaseTemplate>('/cases', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(template),
  });
}

export async function createAgent(
  brainType: BrainType,
  model: string,
): Promise<AgentResponse> {
  return request<AgentResponse>('/agents', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: 'Scratch Demo Agent',
      brain_type: brainType,
      brain_model: model.trim() || null,
      tools: ['calculator', 'final_answer'],
    }),
  });
}

export async function createRun(agentId: string, input: string): Promise<RunResponse> {
  return request<RunResponse>('/agents/' + agentId + '/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input }),
  });
}

export async function executeRun(runId: string): Promise<RunResponse> {
  return request<RunResponse>('/runs/' + runId + '/execute', { method: 'POST' });
}

export async function stepRun(runId: string): Promise<RunResponse> {
  return request<RunResponse>('/runs/' + runId + '/steps', { method: 'POST' });
}

export async function pauseRun(runId: string): Promise<RunResponse> {
  return request<RunResponse>('/runs/' + runId + '/pause', { method: 'POST' });
}

export async function resumeRun(runId: string, input?: string, selectionId?: string, selectionIds?: string[]): Promise<RunResponse> {
  return request<RunResponse>('/runs/' + runId + '/resume', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...(input ? { input } : {}), ...(selectionId ? { selection_id: selectionId } : {}), ...(selectionIds?.length ? { selection_ids: selectionIds } : {}) }),
  });
}

export interface CaseRunResponse {
  case_id: string;
  agent: AgentResponse;
  run: RunResponse;
}

export interface WorkflowRunResponse {
  workflow: WorkflowDocument;
  agent: AgentResponse;
  run: RunResponse;
}

export async function createWorkflowRun(
  workflow: WorkflowDocument,
  input?: string,
): Promise<WorkflowRunResponse> {
  return request<WorkflowRunResponse>('/workflows/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input ? { workflow, input } : { workflow }),
  });
}

export async function createCaseRun(
  caseId: string,
  input?: string,
): Promise<CaseRunResponse> {
  return request<CaseRunResponse>('/cases/' + caseId + '/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input ? { input } : {}),
  });
}

export async function getEvents(runId: string, after = 0): Promise<AgentEvent[]> {
  return request<AgentEvent[]>('/runs/' + runId + '/events?after=' + after);
}

export async function getReplay(runId: string): Promise<ReplayResponse> {
  return request<ReplayResponse>('/runs/' + runId + '/replay');
}

const EVENT_TYPES = [
  'run.start',
  'context.build',
  'context.request',
  'llm.start',
  'llm.end',
  'action.validate',
  'llm.error',
  'tool.start',
  'tool.batch.start',
  'tool.batch.end',
  'tool.parallel.start',
  'tool.parallel.end',
  'tool.end',
  'tool.error',
  'tool.confirmed',
  'tool.rejected',
  'agent.confirmation_required',
  'agent.finish',
  'agent.waiting_user',
  'agent.route',
  'agent.error',
  'run.paused',
  'run.resumed',
  'user.input',
  'user.confirmation',
];

export function openRunEventStream(
  runId: string,
  onEvent: (event: AgentEvent) => void,
  onEnd: () => void,
): EventSource {
  const source = new EventSource(API_BASE + '/runs/' + runId + '/events/stream');
  const handleEvent = (event: Event) => {
    const message = event as MessageEvent<string>;
    try {
      onEvent(JSON.parse(message.data) as AgentEvent);
    } catch {
      // Ignore malformed browser-side frames; the persisted replay remains authoritative.
    }
  };
  EVENT_TYPES.forEach((type) => source.addEventListener(type, handleEvent));
  source.addEventListener('stream.end', () => {
    source.close();
    onEnd();
  });
  source.onerror = () => {
    source.close();
  };
  return source;
}
