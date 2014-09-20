package jrnnlm.utils;

public class Logger {

    public static void println(String s) {

        System.out.println(s);
    }

    public static void info(String s) {

        System.out.print(String.format("<Thread%d> ", Thread.currentThread().getId()));
        System.out.println(s);
    }

    public static void error(String s) {

        System.out.print(String.format("<Thread%d> ", Thread.currentThread().getId()));
        System.out.print("[ERROR] ");
        System.out.println(s);
    }
}
