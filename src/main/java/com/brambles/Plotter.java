package com.brambles;

import java.awt.Color;
import java.awt.Graphics;
import javax.swing.JFrame;
import javax.swing.JPanel;

import static com.brambles.Stuff.Dataset;

public class Plotter extends JPanel {
    private int[][] points;
    private int frameSize;

    public Plotter(int[][] points, int frameSize) {
        this.frameSize = frameSize;
        this.points = points;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        for (int[] point : points) {
            int x = point[0] + frameSize / 2;
            int y = point[1] + frameSize / 2;
            int color = point[2];
            g.setColor(new Color(color));
            g.fillOval(x, y, 5, 5);
        }
    }

    public static void plot(Dataset dataset, int frameSize) {
        int[][] points = new int[dataset.examples.length][];

        for (int i = 0; i < dataset.examples.length; i++) {
            points[i] = new int[]{(int) (dataset.examples[i][0] * 25), (int) (dataset.examples[i][1] * 25), dataset.labels[i][0] == 1.0 ? 0x0000FF : 0xFF0000};
        }

        JFrame frame = new JFrame();
        Plotter plotter = new Plotter(points, frameSize);
        frame.add(plotter);
        frame.setSize(frameSize, frameSize);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}
