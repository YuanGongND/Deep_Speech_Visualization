font_size = 20

%%
diffProposed = csvread( 'diffPro.csv' );
diffProposed = diffProposed( 1, : );
plot(diffProposed, 'linewidth', 3);

set(gcf, 'Position', [0, 0, 800, 600]);
xlabel('Iteration Index','FontSize',font_size);
ylabel('Mean Change Rate of the Filters (%)','FontSize',font_size);
set(gca,'FontSize',font_size);
saveas(gcf,'changeRate_1','epsc');

%%
diffCmp = csvread( 'diffCmp.csv' );
hold off;
for i = 1: 6
    plot(diffCmp(i,:), 'linewidth', 3);
    hold on;
end
ylim( [ 0, 15 ] );
set(gcf, 'Position', [0, 0, 800, 600]);
xlabel('Iteration Index','FontSize',font_size);
ylabel('Mean Change Rate of the Filters (%)','FontSize',font_size);
legend( {'conv1','conv2','conv3','conv4','conv5','conv6'},'Location','northeast','FontSize',font_size )
set(gca,'FontSize',font_size);
saveas(gcf,'changeRate_2','epsc');
hold off;