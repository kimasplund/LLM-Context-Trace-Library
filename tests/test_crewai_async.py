from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lctl.integrations.crewai import CREWAI_AVAILABLE, LCTLCrew, LCTLSession


@pytest.mark.skipif(not CREWAI_AVAILABLE, reason="crewai not installed")
@pytest.mark.asyncio
async def test_kickoff_async_tracing():
    # Setup mocks
    mock_agent = MagicMock()
    mock_agent.role = "tester"
    mock_agent.goal = "test"

    mock_task = MagicMock()
    mock_task.description = "test task"

    # Mock the internal Crew
    with patch("lctl.integrations.crewai.Crew") as MockCrew:  # noqa: N806  # class-like mock name
        mock_crew_instance = MockCrew.return_value
        # Mock kickoff_async to return a result
        mock_crew_instance.kickoff_async = AsyncMock(return_value="Async Result")
        mock_crew_instance.agents = [mock_agent]
        mock_crew_instance.tasks = [mock_task]

        # Initialize LCTLCrew
        lctl_crew = LCTLCrew(agents=[mock_agent], tasks=[mock_task], verbose=True)

        # Spy on the session
        lctl_crew._session = MagicMock(spec=LCTLSession)

        # Execute
        result = await lctl_crew.kickoff_async()

        # Verify result
        assert result == "Async Result"

        # Verify tracing
        # 1. Check if kickoff event was recorded
        lctl_crew._session.step_start.assert_any_call(
            agent="crew-manager",
            intent="kickoff_async",
            input_summary="Starting async sequential crew",
        )

        # 2. IMPORTANT: Verify if callbacks were hooked up.
        # This test checks the *wrapper* logic. Use integration test for actual callback firing.
        # But here we can check if LCTLCrew sets the callbacks on the underlying crew.
        assert mock_crew_instance.step_callback == lctl_crew._on_step
        assert mock_crew_instance.task_callback == lctl_crew._on_task_complete
