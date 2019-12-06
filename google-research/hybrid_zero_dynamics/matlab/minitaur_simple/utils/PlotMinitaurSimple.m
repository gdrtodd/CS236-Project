% Copyright 2019 Google LLC
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     https://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function PlotMinitaurSimple(t_log, q_log, make_vid, tf_plotFrames)

if nargin < 4
  tf_plotFrames = false;
  if nargin < 3
    make_vid.flag = false;
    make_vid.filename = [];
    make_vid.pov = [0, 0];
    make_vid.cycles = 5;
    make_vid.visibility = 'on';
  end
end

%% Smoothen Animation
fs = 100; % frequency
[ts_log, qs_log] = even_sample(t_log, q_log, fs);
ts_log = repmat(ts_log, [1, make_vid.cycles]);
qs_log = repmat(qs_log, [1, make_vid.cycles]);

samples = length(ts_log);

az = [zeros(1, floor(samples/2)), linspace(0, make_vid.pov(1), ...
  (samples - floor(samples/2)))];
el = [zeros(1, floor(samples/2)), linspace(0, make_vid.pov(2), ...
  (samples - floor(samples/2)))];

%% Leg Colors
FRColor = 'r';
FLColor = 'g';
BRColor = 'b';
BLColor = 'y';

%% Chassis colors
chassisColor = 'k';

%% Chassis Dimensions
ln = 0.2375 * 2;
bd = 0.13 * 2;
ht = 0.12;

% plot links
if make_vid.visibility
  h = figure('visible', make_vid.visibility, 'Position', [1, 0, 1920, 1200], ...
    'MenuBar', 'none', 'ToolBar', 'none', 'resize', 'off');
else
  h = figure('visible', make_vid.visibility);
end

%fullscreen(h);
%axes('Units', 'pixels', 'Position', [320, 180, 1140, 810]);
% axis vis3d

xlabel('x');
ylabel('y');
zlabel('z');
title('3D View');


frames = [];
for i = 1:size(qs_log, 2)
  q = real(qs_log(:, i));

  [terrain.Tx, terrain.Ty] = meshgrid(-0.4+q(1):0.3:0.6+q(1), ...
    -0.5+q(2):0.3:0.5+q(2));
  terrain.Tz = 0 .* terrain.Tx;

  if tf_plotFrames
    rBase = r_ChassisCenter(q);

    r_MFLL = r_motor_front_leftL_joint(q);
    r_MFRR = r_motor_front_rightR_joint(q);

    r_MBLL = r_motor_back_leftL_joint(q);
    r_MBRR = r_motor_back_rightR_joint(q);

    r_KFLL = r_knee_front_leftL_joint(q);
    r_KFRR = r_knee_front_rightR_joint(q);

    r_KBLL = r_knee_back_leftL_joint(q);
    r_KBRR = r_knee_back_rightR_joint(q);

    r_TFL = r_FrontLeftToe(q);
    r_TFR = r_FrontRightToe(q);
    r_TBL = r_BackLeftToe(q);
    r_TBR = r_BackRightToe(q);
  end

  pBase = p_ChassisCenter(q);

  p_MFLL = p_motor_front_leftL_joint(q);
  p_MFRR = p_motor_front_rightR_joint(q);

  p_MBLL = p_motor_back_leftL_joint(q);
  p_MBRR = p_motor_back_rightR_joint(q);

  p_KFLL = p_knee_front_leftL_joint(q);
  p_KFRR = p_knee_front_rightR_joint(q);

  p_KBLL = p_knee_back_leftL_joint(q);
  p_KBRR = p_knee_back_rightR_joint(q);

  p_TFL = p_FrontLeftToe(q);
  p_TFR = p_FrontRightToe(q);
  p_TBL = p_BackLeftToe(q);
  p_TBR = p_BackRightToe(q);

  ground = surf(terrain.Tx, terrain.Ty, terrain.Tz);
  hold on;

  pChassis = Rectangle(ln, bd, ht, eul2rotm(q(4:6)', 'XYZ'), pBase, chassisColor);

  % define thighs
  pLink_FLL_HK = line([p_MFLL(1), p_KFLL(1)], [p_MFLL(2), p_KFLL(2)], ...
    [p_MFLL(3), p_KFLL(3)]);
  pLink_FRR_HK = line([p_MFRR(1), p_KFRR(1)], [p_MFRR(2), p_KFRR(2)], ...
    [p_MFRR(3), p_KFRR(3)]);

  pLink_BLL_HK = line([p_MBLL(1), p_KBLL(1)], [p_MBLL(2), p_KBLL(2)], ...
    [p_MBLL(3), p_KBLL(3)]);
  pLink_BRR_HK = line([p_MBRR(1), p_KBRR(1)], [p_MBRR(2), p_KBRR(2)], ...
    [p_MBRR(3), p_KBRR(3)]);

  % define shins
  pLink_FLL_KA = line([p_KFLL(1), p_TFL(1)], [p_KFLL(2), p_TFL(2)], ...
    [p_KFLL(3), p_TFL(3)]);
  pLink_FRR_KA = line([p_KFRR(1), p_TFR(1)], [p_KFRR(2), p_TFR(2)], ...
    [p_KFRR(3), p_TFR(3)]);

  pLink_BLL_KA = line([p_KBLL(1), p_TBL(1)], [p_KBLL(2), p_TBL(2)], ...
    [p_KBLL(3), p_TBL(3)]);
  pLink_BRR_KA = line([p_KBRR(1), p_TBR(1)], [p_KBRR(2), p_TBR(2)], ...
    [p_KBRR(3), p_TBR(3)]);

  set(ground);
  set(pChassis, 'LineWidth', 1.5);

  set(pLink_FLL_HK, 'LineWidth', 4, 'Color', FLColor);
  set(pLink_FRR_HK, 'LineWidth', 4, 'Color', FRColor);
  set(pLink_BLL_HK, 'LineWidth', 4, 'Color', BLColor);
  set(pLink_BRR_HK, 'LineWidth', 4, 'Color', BRColor);


  set(pLink_FLL_KA, 'LineWidth', 3, 'Color', FLColor);
  set(pLink_FRR_KA, 'LineWidth', 3, 'Color', FRColor);
  set(pLink_BLL_KA, 'LineWidth', 3, 'Color', BLColor);
  set(pLink_BRR_KA, 'LineWidth', 3, 'Color', BRColor);

  set(text(0, 0, -0.15, ['Time:', num2str(ts_log(i))]));
  set(text(-0.20, 0, -0.20, ['HT-TFR ', num2str(p_TFR(3)), ' HT-TFL ', ...
    num2str(p_TFL(3))]));
  set(text(-0.20, 0, -0.25, ['HT-TBR ', num2str(p_TBR(3)), ' HT-TBL ', ...
    num2str(p_TBL(3))]));


  % plot frames
  if tf_plotFrames
    plotFrame(q(1:3), q(4:6))
    plotFrame(p_FrontLeftToe(q), r_FrontLeftToe(q));
    plotFrame(p_FrontRightToe(q), r_FrontRightToe(q));
    plotFrame(p_BackLeftToe(q), r_BackLeftToe(q));
    plotFrame(p_BackRightToe(q), r_BackRightToe(q));
  end
  %
  xlim([-0.75 + q(1), 0.75 + q(1)]);
  ylim([-0.5 + q(2), 0.5 + q(2)]);
  zlim([-0.75 + q(3), 0.75 + q(3)]);
  view(az(i), el(i));
  frame = getframe(h);
  frames = [frames, frame];
  if make_vid.flag && contains(make_vid.filename, 'gif')
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    % Write to the GIF File
    if i == 1
      imwrite(imind, cm, make_vid.filename, 'gif', 'Loopcount', inf, ...
        'ScreenSize', [1440, 810], 'DelayTime', (1 / fs));
    else
      imwrite(imind, cm, make_vid.filename, 'gif', 'WriteMode', 'append', ...
        'DelayTime', (1 / fs));
    end
  end

  drawnow;
  hold off

  pause(1e-4)
end % for


if contains(make_vid.filename, 'avi')
  writerObj = VideoWriter(make_vid.filename);
  writerObj.FrameRate = 24;
  writerObj.Quality = 100; % Default 75
  % set the seconds per image
  % open the video writer
  open(writerObj);
  % write the frames to the video
  for i = 1:length(frames)
    % convert the image to a frame
    frame = frames(i);
    writeVideo(writerObj, frame);
  end
  % close the writer object
  close(writerObj);
end

end % function
